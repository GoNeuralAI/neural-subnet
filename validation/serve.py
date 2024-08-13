import os
import io
import torch
import clip
import uvicorn
import argparse
from pydantic import BaseModel
from PIL import Image 
import cv2
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.renderer import (
    MeshRenderer, MeshRasterizer, RasterizationSettings,
    HardPhongShader, PointLights, PerspectiveCameras
)
from pytorch3d.renderer import look_at_view_transform
from torchvision.utils import save_image
from skimage.metrics import structural_similarity as ssim
from torchvision import transforms
import numpy as np
from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.responses import FileResponse
from transformers import CLIPProcessor, CLIPModel

class ValidateRequest(BaseModel):
    prompt: str
    uid: int = 0
    
class ValidateResponse(BaseModel):
    score: float
    
app = FastAPI()

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8094)
    args, extras = parser.parse_known_args()
    return args, extras

args, _ = get_args()

# Global variables
global image_path
image_path = []
global render_params
render_params = {}

# Set up the directory and file paths
DATA_DIR = './results'
OUTPUT_DIR = './output_images'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

device = "cuda" if torch.cuda.is_available() else "cpu"

# Function to load an image and prepare for CLIP
def load_image(image_buffer):
    image = Image.open(image_buffer).convert("RGB")
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize the image to match the input size expected by the model
        transforms.ToTensor(),          # Convert the image to a tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize the image
    ])
    return preprocess(image).unsqueeze(0).to(device)
    
def render_mesh(obj_file: str, distance: float = 3.0, elevation: float = 20.0, azimuth: float = 0.0, 
                image_size: int = 512, angle_step: int = 30):
    global image_path
    render_images = []
    before_render = []
    try:
        # Load the mesh
        mesh = load_objs_as_meshes([obj_file], device=device)
        
        # Renderer setup
        R, T = look_at_view_transform(distance, elevation, azimuth)
        cameras = PerspectiveCameras(device=device, R=R, T=T)
        raster_settings = RasterizationSettings(
            image_size=image_size,
            blur_radius=0.0,
            faces_per_pixel=1,
        )
        lights = PointLights(device=device, location=[[2.0, 2.0, 2.0]])
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
        angles = range(0, 360, angle_step)  # Angles from 0 to 330 degrees with step size
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
            
            # Ensure the image tensor is in the [0, 1] range
            image = (image - image.min()) / (image.max() - image.min())
            
            # Create a black background
            black_background = torch.zeros_like(image)
            
            # Composite the image with the black background
            alpha = images[0, ..., 3].cpu().detach()  # Extract the alpha channel
            alpha = alpha.unsqueeze(0).expand_as(image)  # Match the shape of the image
            image = image * alpha + black_background * (1 - alpha)
            
            ndarr = image.mul(255).clamp(0, 255).byte().numpy().transpose(1, 2, 0)  # Convert to [H, W, C]
            pil_image = Image.fromarray(ndarr)
            buffer = io.BytesIO()
            pil_image.save(buffer, format='PNG')  # Save the PIL image to the buffer
            buffer.seek(0)
            before_render.append(buffer)
            
            loaded_image = load_image(buffer)
            render_images.append(loaded_image)
            
        return render_images, before_render
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def render(
    prompt_image: str,
    id: int = 1
):
    global render_params
    
    print(f"promt_image: {prompt_image} : id={id}")
    # Store parameters
    render_params = {
        "prompt_image": prompt_image,
        "preview_image_path": os.path.join(DATA_DIR, f"{id}/preview.png")
    }

    # Use the uploaded objective file for rendering
    obj_file = os.path.join(DATA_DIR, f"{id}/output.obj")

    # Print the file size
    image_files = render_mesh(obj_file)
    
    print(f"output mesh:{len(image_files)}")
    if not image_files:
        raise HTTPException(status_code=500, detail="Rendering failed")
    
    return image_files

@app.get("/download/{angle}")
def download_image(angle: int):
    image_filename = os.path.join(OUTPUT_DIR, f'image_{angle}.png')
    if not os.path.exists(image_filename):
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(image_filename)

@app.post("/validate/")
async def validate(data: ValidateRequest) -> ValidateResponse:
    prompt = data.prompt
    uid = data.uid
    
    global render_params
    try:
        rendered_images, before_images = await render(prompt_image=prompt, id=uid)
        print(len(rendered_images))
        preview_image_path = render_params.get("preview_image_path")

        # Load the cuda & CLIP model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, preprocess = clip.load("ViT-B/32", device=device)

        # Image processing

        # Load all rendered images
        print("--------start render images-------------")
        preview_image = load_image(preview_image_path)

        print("--------end rendered images-------------")

        # Function to compute similarity using CLIP
        def compute_clip_similarity(image1, image2):
            with torch.no_grad():
                image1_features = model.encode_image(image1)
                image2_features = model.encode_image(image2)
                similarity = torch.nn.functional.cosine_similarity(image1_features, image2_features).item()
            return similarity

        def compute_clip_similarity_prompt(text, image_path):
            # Load the model and processor
            model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32')
            processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
            
            # Preprocess the inputs
            image = Image.open(image_path)  # Change to your image path        
            image_inputs = processor(images=image, return_tensors="pt")
            text_inputs = processor(text=text, return_tensors="pt", truncation=True)
            
            # Get the embeddings
            image_embeddings = model.get_image_features(**image_inputs)
            text_embeddings = model.get_text_features(**text_inputs)
            
            # Normalize the embeddings to unit length
            image_embeddings = image_embeddings / image_embeddings.norm(dim=1, keepdim=True)
            text_embeddings = text_embeddings / text_embeddings.norm(dim=1, keepdim=True)
            with torch.no_grad():
                similarity = torch.nn.functional.cosine_similarity(text_embeddings, image_embeddings).item()
            return similarity
        
        S0 = compute_clip_similarity_prompt(prompt, preview_image_path)

        print(f"similarity: {S0}")

        Si = [compute_clip_similarity(preview_image, img) for img in rendered_images]
        print(f"similarities: {Si}")

        def resize_image(image, target_size=(256, 256)):
            """Resize an image to the target size."""
            return image.resize(target_size, Image.Resampling.LANCZOS)

        def pil_to_cv(image):
            """Convert PIL Image to OpenCV format and resize."""
            image = resize_image(image)  # Resize image to ensure dimensions match
            return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        if rendered_images and before_images:
            Q0 = ssim(pil_to_cv(Image.open(preview_image_path)), pil_to_cv(Image.open(before_images[0])), win_size=3)
            Qi = [ssim(pil_to_cv(Image.open(preview_image_path)), pil_to_cv(Image.open(before_image)), win_size=3) for before_image in before_images]
        else:
            Q0 = 0  # No comparison available, set to 0 or an appropriate value indicating no data
            Qi = []

        S_geo = np.exp(np.log(Si).mean())
        Q_geo = np.exp(np.log(Qi).mean())

        # Total Similarity Score (Stotal)
        S_total = S0 * S_geo + Q0 * Q_geo

        print(S_total)

        return ValidateResponse(
            score=S_total
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Uncomment this section to run the FastAPI app locally
if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=args.port)
