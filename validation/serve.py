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
import numpy as np
from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.responses import FileResponse
from transformers import CLIPProcessor, CLIPModel
from models import ValidateRequest, ValidateResponse
from rendering import render, load_image

app = FastAPI()

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8094)
    args, extras = parser.parse_known_args()
    return args, extras

args, _ = get_args()

# Set up the directory and file paths
DATA_DIR = './results'
OUTPUT_DIR = './output_images'

device = "cuda" if torch.cuda.is_available() else "cpu"

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
    
    try:
        rendered_images, before_images = await render(prompt_image=prompt, id=uid)
        print(len(rendered_images))
        preview_image_path = os.path.join(DATA_DIR, f"{uid}/preview.png")

        # Load the cuda & CLIP model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, preprocess = clip.load("ViT-B/32", device=device)

        # Load all rendered images
        
        preview_image = load_image(preview_image_path)
        print("prev_image successfully")

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
