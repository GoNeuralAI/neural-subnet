import os
import argparse
import numpy as np
import torch
import rembg
import time
import random
import zipfile
import uvicorn
from io import BytesIO
from PIL import Image
from torchvision.transforms import v2
from pytorch_lightning import seed_everything
from omegaconf import OmegaConf
from einops import rearrange, repeat
from tqdm import tqdm
from huggingface_hub import hf_hub_download
from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler

from src.utils.train_util import instantiate_from_config
from fastapi import FastAPI, HTTPException, Body
from fastapi.responses import StreamingResponse
from src.utils.camera_util import (
    FOV_to_intrinsics, 
    get_zero123plus_input_cameras,
    get_circular_camera_poses,
    get_render_cameras
)
from src.utils.mesh_util import save_obj, save_obj_with_mtl
from src.utils.infer_util import remove_background, resize_foreground, save_video, render_frames


###############################################################################
# Arguments.
###############################################################################

parser = argparse.ArgumentParser()
parser.add_argument('config', type=str, help='Path to config file.')
# parser.add_argument('input_path', type=str, help='Path to input image or directory.')
parser.add_argument("--port", type=int, default=8093)
parser.add_argument('--output_path', type=str, default='outputs/', help='Temporary output directory.')
parser.add_argument('--diffusion_steps', type=int, default=75, help='Denoising Sampling steps.')
parser.add_argument('--seed', type=int, default=42, help='Random seed for sampling.')
parser.add_argument('--scale', type=float, default=1.0, help='Scale of generated object.')
parser.add_argument('--distance', type=float, default=4.5, help='Render distance.')
parser.add_argument('--view', type=int, default=6, choices=[4, 6], help='Number of input views.')
parser.add_argument('--no_rembg', action='store_true', help='Do not remove input background.')
parser.add_argument('--export_texmap', action='store_true', help='Export a mesh with texture map.')
parser.add_argument('--save_video', action='store_true', help='Save a circular-view video.')
args = parser.parse_args()
seed_everything(args.seed)


app = FastAPI()

###############################################################################
# Stage 0: Configuration.
###############################################################################

config = OmegaConf.load(args.config)
config_name = os.path.basename(args.config).replace('.yaml', '')
model_config = config.model_config
infer_config = config.infer_config

IS_FLEXICUBES = True if config_name.startswith('instant-mesh') else False

device = torch.device('cuda')

# make output directories
image_path = os.path.join(args.output_path, config_name, 'images')
mesh_path = os.path.join(args.output_path, config_name, 'meshes')
video_path = os.path.join(args.output_path, config_name, 'videos')
os.makedirs(image_path, exist_ok=True)
os.makedirs(mesh_path, exist_ok=True)
os.makedirs(video_path, exist_ok=True)

# load playground v2.5 model
pg_pipeline = DiffusionPipeline.from_pretrained(
    "playgroundai/playground-v2.5-1024px-aesthetic",
    torch_dtype=torch.float16,
    variant="fp16",
    ).to("cuda")

# create remove_background session
rembg_session = rembg.new_session()

# load diffusion model
print('Loading diffusion model ...')
zero_pipeline = DiffusionPipeline.from_pretrained(
    "sudo-ai/zero123plus-v1.2", 
    custom_pipeline="zero123plus",
    torch_dtype=torch.float16,
)
zero_pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
    zero_pipeline.scheduler.config, timestep_spacing='trailing'
)

# load custom white-background UNet
print('Loading custom white-background unet ...')
if os.path.exists(infer_config.unet_path):
    unet_ckpt_path = infer_config.unet_path
else:
    unet_ckpt_path = hf_hub_download(repo_id="TencentARC/InstantMesh", filename="diffusion_pytorch_model.bin", repo_type="model")
state_dict = torch.load(unet_ckpt_path, map_location='cpu')
zero_pipeline.unet.load_state_dict(state_dict, strict=True)

zero_pipeline = zero_pipeline.to(device)

# load reconstruction model
print('Loading reconstruction model ...')
mesh_model = instantiate_from_config(model_config)
if os.path.exists(infer_config.model_path):
    model_ckpt_path = infer_config.model_path
else:
    model_ckpt_path = hf_hub_download(repo_id="TencentARC/InstantMesh", filename=f"{config_name.replace('-', '_')}.ckpt", repo_type="model")
state_dict = torch.load(model_ckpt_path, map_location='cpu')['state_dict']
state_dict = {k[14:]: v for k, v in state_dict.items() if k.startswith('lrm_generator.')}
mesh_model.load_state_dict(state_dict, strict=True)


mesh_model = mesh_model.to(device)
if IS_FLEXICUBES:
    mesh_model.init_flexicubes_geometry(device, fovy=30.0)
mesh_model = mesh_model.eval()

### text-to-mesh generation endpoint
@app.post("/generate_from_text/")
async def generate_mesh(prompt: str = Body()):
    
    print(prompt)
    # generate image with text-to-image model
    main_image = await _generate_image(prompt)
    
    # generate preview image from the main image
    prev_images = await _generate_preview(main_image)

    # generate mesh object from preview images
    mesh_obj = await _generate_mesh(prev_images)

    prev_path_idx = os.path.join(image_path, f'preview.png')
    mesh_path_idx = os.path.join(mesh_path, f'output.obj')
    mtl_path_idx = os.path.join(mesh_path, f'output.mtl')
    texture_path_idx = os.path.join(mesh_path, f'output.png')
    
    print("loading...")
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
        zip_file.writestr('preview.png', open(prev_path_idx, 'rb').read())
        zip_file.writestr('output.obj', open(mesh_path_idx, 'r').read())
        zip_file.writestr('output.mtl', open(mtl_path_idx, 'r').read())
        zip_file.writestr('output.png', open(texture_path_idx, 'rb').read())
    
    zip_buffer.seek(0)  # Move to the beginning of the BytesIO buffer

    print("Files prepared for download.")

    return StreamingResponse(zip_buffer, media_type='application/zip', headers={"Content-Disposition": "attachment; filename=mesh_files.zip"})
    
    # return StreamingResponse(BytesIO(obj_data.encode()), media_type="application/octet-stream") 


### image-to-mesh generation endpoint
@app.post("/generate_from_image")
async def generate_preview(input_image):
    # generate preview image from the input image
    prev_images = _generate_preview(input_image)

    # generate mesh object from preview images
    mesh_obj = _generate_mesh(prev_images)
        
    return mesh_obj

# Generate image from prompt using playground v2.5 model
async def _generate_image(prompt: str):
    print('Generating main image')
    image = pg_pipeline(
        prompt=prompt, 
        num_inference_steps=50, 
        guidance_scale=3
        ).images[0]
    print('Main image generated')
    
    return image

# Generate preview images with zero123plus model
async def _generate_preview(input_image):
    print('Generating preview images')
    input_image.save(os.path.join(image_path, 'preview.png'))
    input_image = remove_background(input_image, rembg_session)
    
    input_image = resize_foreground(input_image, 0.85)

    prev_image = zero_pipeline(input_image, num_inference_step=args.diffusion_steps).images[0]
    prev_images = np.asarray(prev_image, dtype=np.float32) / 255.0
    prev_images = torch.from_numpy(prev_images).permute(2, 0, 1).contiguous().float()     # (3, 960, 640)
    prev_images = rearrange(prev_images, 'c (n h) (m w) -> (n m) c h w', n=3, m=2)        # (6, 3, 320, 320)
    # prev_image.save(os.path.join(image_path, 'preview.png'))
    print('Preview images generated')

    return prev_images

# Generate mesh object from preview images
async def _generate_mesh(input_images):
    print('Generating mesh object...')
    input_cameras = get_zero123plus_input_cameras(batch_size=1, radius=4.0*args.scale).to(device)
    chunk_size = 20 if IS_FLEXICUBES else 1
    input_images = input_images.unsqueeze(0).to(device)
    input_images = v2.functional.resize(input_images, 320, interpolation=3, antialias=True).clamp(0, 1)
    if args.view == 4:
        indices = torch.tensor([0, 2, 4, 5]).long().to(device)
        input_images = input_images[:, indices]
        input_cameras = input_cameras[:, indices]

    with torch.no_grad():
        # get triplane
        planes = mesh_model.forward_planes(input_images, input_cameras)
        print('Mesh object generated...')

        # get mesh
        mesh_path_idx = os.path.join(mesh_path, f'output.obj')

        mesh_out = mesh_model.extract_mesh(
            planes,
            use_texture_map=args.export_texmap,
            **infer_config,
        )

        # Generate texture map
        if args.export_texmap:
            vertices, faces, uvs, mesh_tex_idx, tex_map = mesh_out
            save_obj_with_mtl(
                vertices.data.cpu().numpy(),
                uvs.data.cpu().numpy(),
                faces.data.cpu().numpy(),
                mesh_tex_idx.data.cpu().numpy(),
                tex_map.permute(1, 2, 0).data.cpu().numpy(),
                mesh_path_idx,
            )
        else:
            vertices, faces, vertex_colors = mesh_out
            save_obj(vertices, faces, vertex_colors, mesh_path_idx)
        print(f"Mesh saved to {mesh_path_idx}")

        # # Generate video
        # if args.save_video:
        #     video_path_idx = os.path.join(video_path, f'output.mp4')
        #     render_size = infer_config.render_resolution
        #     render_cameras = get_render_cameras(
        #         batch_size=1, 
        #         M=120, 
        #         radius=args.distance, 
        #         elevation=20.0,
        #         is_flexicubes=IS_FLEXICUBES,
        #     ).to(device)
            
        #     frames = render_frames(
        #         mesh_model, 
        #         planes, 
        #         render_cameras=render_cameras, 
        #         render_size=render_size, 
        #         chunk_size=chunk_size, 
        #         is_flexicubes=IS_FLEXICUBES,
        #     )

        #     save_video(
        #         frames,
        #         video_path_idx,
        #         fps=30,
        #     )
        #     print(f"Video saved to {video_path_idx}")

    return

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=args.port)
