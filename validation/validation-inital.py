import os
import io
import torch
import clip
import uvicorn
import time
import argparse
from PIL import Image 
import cv2
from skimage.metrics import structural_similarity as ssim
import numpy as np
from fastapi import FastAPI, HTTPException, File, Form
from models import ValidateRequest, ValidateResponse
from rendering import render, load_image
import torchvision.transforms as transforms
import open_clip

app = FastAPI()

# Load the cuda & CLIP model
device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

try:
    model, preprocess = clip.load("ViT-B/32", device=device)
    open_model, _, open_preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
    open_model.eval()  # model in train mode by default, impacts some models with BatchNorm or stochastic depth active
    tokenizer = open_clip.get_tokenizer('ViT-B-32')
    # q_model = torch.hub.load(repo_or_dir="miccunifi/QualiCLIP", source="github", model="QualiCLIP").to(device)
    # q_model.eval().to(device)
except Exception as e:
    print("load model error")
    
print("mdoel")
    
# Define the preprocessing pipeline
q_preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
])

def exp_f(x):
     return 13 * x ** 3

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

# def compute_quality(image_path):
#     img = Image.open(image_path).convert("RGB")
#     img = q_preprocess(img).unsqueeze(0).to(device)
#     with torch.no_grad():
#         score = q_model(img)
#     return score.item()

def compute_clip_similarity_prompt(text_prompt, image_path):
    global open_model, open_preprocess
    # Preprocess the inputs
    image = Image.open(image_path)  # Change to your image path
    image = open_preprocess(image).unsqueeze(0)
    
    labels_list = [text_prompt]
    text = tokenizer(labels_list)
    
    with torch.no_grad():
        image_features = open_model.encode_image(image)
        text_features = open_model.encode_text(text)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        text_probs = (image_features @ text_features.T)

    return exp_f(text_probs.item())

args, _ = get_args()

# Set up the directory and file paths
DATA_DIR = './results'

@app.post("/validate/")
async def validate(data: ValidateRequest) -> ValidateResponse:
    prompt = data.prompt
    uid = data.uid
    start_time = time.time()
    try:
        rendered_images, before_images = render(prompt=prompt, id=uid)
        # print(f"render time: {time.time() - start_time}")
        preview_image_path = os.path.join(DATA_DIR, f"{uid}/preview.png")

        preview_image = load_image(preview_image_path)
        # print(f"load model time: {time.time() - start_time}")

        # Q0 = compute_quality(preview_image_path)
        print(f"load model time: {time.time() - start_time}")
        
        # if Q0 > 0.4:
        S0 = compute_clip_similarity_prompt(prompt, preview_image_path)
        # else:
        #     S0 = 0
            
        print(f"----- S0 similarity: {S0} ----------- Rendered Time : {time.time() - start_time}s")

        Si = [compute_clip_similarity(preview_image, img) for img in rendered_images]
        
        Ri = [compute_clip_similarity_prompt(prompt, before_image) for before_image in before_images]
        
        ESi = [compute_clip_similarity(rendered_images[0], img) for img in rendered_images]
        
        # Qi = [compute_quality(img) for img in rendered_images]

        S_geo = np.exp(np.log(Si).mean())
        R_geo = np.exp(np.log(Ri).mean())
        ES_geo = np.exp(np.log(ESi).mean())
        # Q_geo = np.exp(np.log(Qi).mean())
        
        print("--------- Rendered images similarities with preview image --------")
        print(Si)
        print(f"S_geo: {S_geo}")
        
        print("********* Rendered images similarities with first rendered image **********")
        print(ESi)
        print(f"ES_geo: {ES_geo}")
        
        print("--------- Rendered images similarities with text prompt ---------")
        print(Ri)
        print(f"R_geo: {R_geo}")
        
        print("--------- Images quality ---------")
        # print(Qi)
        # print(f"R_geo: {Q_geo}")
        
        print(time.time() - start_time)

        # Total Similarity Score (Stotal)
        S_total = S0 * 0.3 + S_geo * 0.3 + R_geo * 0.4
        
        print(f"S_total: {S_total}")

        return ValidateResponse(
            score=S_total
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Uncomment this section to run the FastAPI app locally
if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=args.port)
