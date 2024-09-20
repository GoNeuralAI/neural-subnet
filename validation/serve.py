import os
import io
import torch
import clip
import uvicorn
import time
import argparse
from pydantic import BaseModel
from PIL import Image 
import cv2
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.renderer import (
    MeshRenderer, MeshRasterizer, RasterizationSettings,
    HardPhongShader, PointLights, PerspectiveCameras
)
from skimage.metrics import structural_similarity as ssim
import numpy as np
from fastapi import FastAPI, HTTPException, File, Form
from transformers import CLIPProcessor, CLIPModel
from models import ValidateRequest, ValidateResponse
from rendering import render, load_image
import open_clip

app = FastAPI()

# Load the cuda & CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
try:
    model, preprocess = clip.load("ViT-B/32", device=device)
    open_model, _, open_preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
    open_model.eval()  # model in train mode by default, impacts some models with BatchNorm or stochastic depth active
    tokenizer = open_clip.get_tokenizer('ViT-B-32')
except Exception as e:
    print("load model error")

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8094)
    args, extras = parser.parse_known_args()
    return args, extras

def resize_image(image, target_size=(256, 256)):
    """Resize an image to the target size."""
    return image.resize(target_size, Image.Resampling.LANCZOS)

def pil_to_cv(image):
    """Convert PIL Image to OpenCV format and resize."""
    image = resize_image(image)  # Resize image to ensure dimensions match
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

def calculate_image_entropy(image):
    """Calculate the entropy of an image."""
    image = resize_image(image)  # Resize image to ensure dimensions match
    histogram = np.histogram(image, bins=256, range=[0, 256])[0]
    histogram_normalized = histogram / histogram.sum()
    histogram_normalized = histogram_normalized[histogram_normalized > 0]  # Remove zeros
    entropy = -np.sum(histogram_normalized * np.log2(histogram_normalized))
    return entropy

def compute_clip_similarity(image1, image2):
    global model
    with torch.no_grad():
        image1_features = model.encode_image(image1)
        image2_features = model.encode_image(image2)
        similarity = torch.nn.functional.cosine_similarity(image1_features, image2_features).item()
    return similarity

def compute_clip_similarity_prompt(text_prompt, image_path):
    global open_model, open_preprocess
    # Preprocess the inputs
    image = Image.open(image_path)  # Change to your image path
    image = open_preprocess(image).unsqueeze(0)
    
    labels_list = [text_prompt]
    text = tokenizer(labels_list)
    
    with torch.no_grad(), torch.cuda.amp.autocast():
        image_features = open_model.encode_image(image)
        text_features = open_model.encode_text(text)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        text_probs = (image_features @ text_features.T)

    return text_probs.item()

args, _ = get_args()

# Set up the directory and file paths
DATA_DIR = './results'

device = "cuda" if torch.cuda.is_available() else "cpu"

@app.post("/validate/")
async def validate(data: ValidateRequest) -> ValidateResponse:
    prompt = data.prompt
    uid = data.uid
    start_time = time.time()
    try:
        rendered_images, before_images = await render(prompt_image=prompt, id=uid)
        # print(f"render time: {time.time() - start_time}")
        preview_image_path = os.path.join(DATA_DIR, f"{uid}/preview.png")

        preview_image = load_image(preview_image_path)
        # print(f"load model time: {time.time() - start_time}")

        S0 = compute_clip_similarity_prompt(prompt, preview_image_path)
        ES0 = compute_clip_similarity_prompt(prompt, before_images[0])
        print(f"----- S0 similarity: {S0}")
        print(f"***** ES0 similarity: {ES0}")
        
        if S0 < 0.25:
            return ValidateResponse(
                score=0
            )

        Si = [compute_clip_similarity(preview_image, img) for img in rendered_images]
        
        
        Ri = [compute_clip_similarity_prompt(prompt, before_image) for before_image in before_images]
        
        
        ESi = [compute_clip_similarity(rendered_images[0], img) for img in rendered_images]
        
        
        print(f"S calc time: {time.time() - start_time}")
        
        if rendered_images and before_images:
            # Q0 = calculate_image_entropy(Image.open(preview_image_path))
            Qi = [calculate_image_entropy(Image.open(before_image)) for before_image in before_images]
        else:
            # Q0 = 0  # No comparison available, set to 0 or an appropriate value indicating no data
            Qi = []
            
        print(f"Qi: {Qi}")
        print(f"Qi time: {time.time() - start_time}")
        epsilon = 1e-10
        Qi_array = np.array(Qi)
        Qi_with_epsilon = Qi_array + epsilon

        S_geo = np.exp(np.log(Si).mean())
        R_geo = np.exp(np.log(Ri).mean())
        ES_geo = np.exp(np.log(ESi).mean())
        Q_geo = np.exp(np.log(Qi_with_epsilon).mean())
        
        print("--------- Rendered images similarities with preview image --------")
        print(Si)
        print(f"S_geo: {S_geo}")
        
        print("********* Rendered images similarities with first rendered image **********")
        print(ESi)
        print(f"ES_geo: {ES_geo}")
        
        print("--------- Rendered images similarities with text prompt ---------")
        print(Ri)
        print(f"R_geo: {R_geo}")
        

        # Total Similarity Score (Stotal)
        S_total = S0 * 0.25 + S_geo * 0.3 + R_geo * 0.35
        
        # Exploit Total Similarity Score (Stotal)
        ES_total = S0 * 0.25 + ES_geo * 0.3 + R_geo * 0.35

        print(f"S_total: {S_total}")
        print(f"ES_total: {ES_total}")

        return ValidateResponse(
            score=S_total
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Uncomment this section to run the FastAPI app locally
if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=args.port)
