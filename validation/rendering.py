import os
import io
import torch
from PIL import Image 
import trimesh
from pytorch3d.renderer import (
    MeshRenderer, MeshRasterizer, RasterizationSettings,
    HardPhongShader, PointLights, PerspectiveCameras
)
from pytorch3d.renderer import look_at_view_transform
from torchvision.utils import save_image
from skimage.metrics import structural_similarity as ssim
import numpy as np
from pytorch3d.structures import Meshes
from fastapi import  HTTPException
from torchvision import transforms
from pytorch3d.renderer import TexturesUV, TexturesVertex

DATA_DIR = './validation/results'
OUTPUT_DIR = './validation/output_images'

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


device = "cuda" if torch.cuda.is_available() else "cpu"

def load_glb_as_mesh(glb_file, device='cpu'):
    # Load the .glb file using trimesh
    mesh = trimesh.load(glb_file, file_type='glb', force="mesh")
    
    # Extract vertices, faces, and UV coordinates from the trimesh object
    vertices = torch.tensor(mesh.vertices, dtype=torch.float32, device=device)
    faces = torch.tensor(mesh.faces, dtype=torch.int64, device=device)
    
    # Check if the mesh has texture information
    if hasattr(mesh.visual, 'uv') and mesh.visual.uv is not None:
        # Extract UV coordinates
        uv_coords = torch.tensor(mesh.visual.uv, dtype=torch.float32, device=device)
        
        # Extract texture image
        texture_image = mesh.visual.material.baseColorTexture
        texture_image = np.array(texture_image)
        texture_image = torch.tensor(texture_image, dtype=torch.float32, device=device) / 255.0  # Normalize to [0, 1]
        
        # Create TexturesUV object
        textures = TexturesUV(maps=[texture_image], faces_uvs=[faces], verts_uvs=[uv_coords])
    elif hasattr(mesh.visual, 'vertex_colors'):
        vertex_colors = mesh.visual.vertex_colors
        vertex_colors = torch.tensor(vertex_colors, dtype=torch.float32, device=device) / 255.0
        # Separate RGB and Alpha
        vertex_colors_rgb = vertex_colors[:, :3]
        vertex_colors_alpha = vertex_colors[:, 3].unsqueeze(-1)
        # Apply alpha to RGB
        vertex_colors = vertex_colors_rgb * vertex_colors_alpha
        textures = TexturesVertex(verts_features=[vertex_colors[:, :3]])
    else:
        # Fallback to a simple white texture if no texture is found
        textures = TexturesUV(maps=[torch.ones((1, 1, 3), dtype=torch.float32, device=device)], 
                              faces_uvs=[faces], verts_uvs=[torch.zeros_like(vertices[:, :2])])
    
    # Create a PyTorch3D Meshes object
    pytorch3d_mesh = Meshes(verts=[vertices], faces=[faces], textures=textures)
    return pytorch3d_mesh

# Function to load an image and prepare for CLIP
def load_image(image_buffer):
    image = Image.open(image_buffer).convert("RGB")
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize the image to match the input size expected by the model
        transforms.ToTensor(),          # Convert the image to a tensor
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize the image
    ])
    return preprocess(image).unsqueeze(0).to(device)


def render_mesh(obj_file: str, distance: float = 0.75, elevation: float = 10, azimuth: float = 0.0, 
                image_size: int = 200, angle_step: int = 48):
    render_images = []
    before_render = []
    try:
        # Load the mesh
        print(obj_file)
        mesh = load_glb_as_mesh(glb_file=obj_file, device=device)
        print("The mesh is ", mesh)
        
        # Renderer setup
        R, T = look_at_view_transform(distance, elevation, azimuth, at=((0, 0, 1),))
        cameras = PerspectiveCameras(device=device, R=R, T=T)
        raster_settings = RasterizationSettings(
            image_size=image_size,
            blur_radius=0.0,
            faces_per_pixel=1,
        )
        lights = PointLights(
            device=device,
            location=[[2,2,2]],
            ambient_color=[[1, 1, 1]],  # Standard ambient color
            specular_color=[[0, 0, 0]]  # Standard specular color
        )
        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=cameras,
                raster_settings=raster_settings
            ),
            shader=HardPhongShader(
                device=device,
                cameras=cameras,
                lights=lights
            )
        )

        # Generate and save images
        angles = range(24, 360, angle_step)  # Angles from 0 to 330 degrees with step size

        rendered_image_paths = []

        for angle in angles:
            R, T = look_at_view_transform(distance, elevation, angle)
            cameras = PerspectiveCameras(device=device, R=R, T=T)
            renderer = MeshRenderer(
                rasterizer=MeshRasterizer(
                    cameras=cameras,
                    raster_settings=raster_settings
                ),
                shader=HardPhongShader(
                    device=device,
                    cameras=cameras,
                    lights=lights
                )
            )
            # Render the image
            images = renderer(mesh)
            
            # Ensure the image tensor is in the correct format
            image = images[0, ..., :3].cpu().detach()
            if image.shape[0] != 3:
                image = image.permute(2, 0, 1)  # Change shape to [C, H, W]
            
            # Extract the alpha channel
            alpha = images[0, ..., 3].cpu().detach()  # Extract the alpha channel
            alpha = alpha.unsqueeze(0)  # Match the shape of the image
            
            # Ensure the image tensor is in the [0, 1] range
            image = (image - image.min()) / (image.max() - image.min())

            image_filename = os.path.join(OUTPUT_DIR, f'image_{angle}.jpeg')

            rendered_image_paths.append(str(image_filename))

            ndarr = image.mul(255).clamp(0, 255).byte().numpy().transpose(1, 2, 0)  # Convert to [H, W, C] format
            pil_images = Image.fromarray(ndarr, 'RGB')  # Specify 'RGB' since we don't need alpha for JPEG
            pil_images.save(image_filename, format='JPEG')  # Save as JPEG

            # Composite the image with transparency
            image = torch.cat([image, alpha], dim=0)  # Add alpha channel to the image
            
            # Save rendered images locally
            # image_filename = os.path.join(OUTPUT_DIR, f'image_{angle}.jpeg')
            # pil_image.convert("RGB").save(image_filename, format='JPEG') # save images
            
            # Convert to [H, W, C] format with RGBA
            ndarr = image.mul(255).clamp(0, 255).byte().numpy().transpose(1, 2, 0)
            pil_image = Image.fromarray(ndarr, 'RGBA')  # Specify 'RGBA' to include alpha
            
            buffer = io.BytesIO()
            pil_image.save(buffer, format='PNG')  # Save the PIL image to the buffer
            buffer.seek(0)
            before_render.append(buffer)
            
            loaded_image = load_image(buffer)
            render_images.append(loaded_image)
            
        print("The length of render_images is ", len(render_images))
            
        return render_images, before_render, rendered_image_paths
    except Exception as e:
        print("Erorr occurs during the Rendering meshing, Incorrect formatted file")
        raise HTTPException(status_code=500, detail=str(e))

def render(
    prompt: str,
    id: int = 1,
    verbose: bool = True,
):
    print(f"prompt: {prompt} : id={id}")
    
    # Use the uploaded objective file for rendering
    if verbose:
        obj_file = os.path.join(DATA_DIR, f"{id}/output.glb")
    else:
        obj_file = os.path.join(DATA_DIR, f"{id}/mesh.glb")

    # Print the file size
    image_files = render_mesh(obj_file)
    
    print(f"output mesh:{len(image_files)}")
    if not image_files:
        print("Failed during the file rendering, Incorrect formatting files")
        raise HTTPException(status_code=500, detail="Rendering failed")
    
    return image_files
