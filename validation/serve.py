import os
import torch
import clip
import uvicorn
import argparse
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

app = FastAPI()

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=10025)
    args, extras = parser.parse_known_args()
    return args, extras

args, _ = get_args()

# Global variables
global image_path
image_path = []
global render_params
render_params = {}

# Set up the directory and file paths
DATA_DIR = './files'
OUTPUT_DIR = './output_images'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

device = "cuda" if torch.cuda.is_available() else "cpu"
    
def render_mesh(obj_file: str, distance: float = 3.0, elevation: float = 20.0, azimuth: float = 0.0, 
                image_size: int = 512, angle_step: int = 30):
    global image_path
    image_path = []
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
            
            # Save the image
            image_filename = os.path.join(OUTPUT_DIR, f'image_{angle}.png')
            save_image(image, image_filename)  # Save image
            print(f'Saved image to {image_filename}')
            image_path.append(image_filename)
        return image_path
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def render(
    prompt_image: str = Form(..., description="Prompt Image or Text (as a string)"),
    # preview_image: UploadFile = File(..., description="Preview Image"),
    # objective_file: UploadFile = File(..., description="Objective File"),
    # mtl_file: UploadFile = File(..., description=".MTL File"),
    # png_file: UploadFile = File(..., description="PNG File (Embedding)")
):
    global render_params

    # Store parameters
    render_params = {
        "prompt_image": prompt_image,
        "preview_image_path": os.path.join(DATA_DIR, preview_image.filename)
    }

    # Use the uploaded objective file for rendering
    obj_file = os.path.join(DATA_DIR, objective_file.filename)
    image_files = render_mesh(obj_file)
    if not image_files:
        raise HTTPException(status_code=500, detail="Rendering failed")
    
    return {"image_files": image_files}

@app.get("/download/{angle}")
def download_image(angle: int):
    image_filename = os.path.join(OUTPUT_DIR, f'image_{angle}.png')
    if not os.path.exists(image_filename):
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(image_filename)

@app.post("/validate")
def validate():
    render()
    global render_params
    try:
        output_folder=OUTPUT_DIR
        prompt=render_params.get("prompt_image")
        preview_image_path = render_params.get("preview_image_path")

        # Load the cuda & CLIP model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, preprocess = clip.load("ViT-B/32", device=device)

        # Image processing
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize the image to match the input size expected by the model
            transforms.ToTensor(),          # Convert the image to a tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize the image
        ])


        # Function to load an image and prepare for CLIP
        def load_image(image_path):
            image = Image.open(image_path).convert("RGB")
            return preprocess(image).unsqueeze(0).to(device)

        # Function to load an image and prepare for CLIP
        def load_image(image_path):
            image = Image.open(image_path).convert("RGB")
            return preprocess(image).unsqueeze(0).to(device)


        # Retrieve all image file paths from the output folder
        rendered_images_paths = [os.path.join(output_folder, f) for f in os.listdir(output_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]

        # Load all rendered images
        rendered_images = [load_image(path ) for path in rendered_images_paths]
        preview_image = load_image(preview_image_path)
        rendered_images = [load_image(path) for path in rendered_images_paths]

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

        # Prompt to Preview Image Similarity (S0)
        # For text-to-3D generation, the similarity between the textual prompt and the preview image is calculated using a CLIP model.
        # Formula:    S_0=CLIP_similarity ( Prompt,Preview Image )

        # if the prompt is text
        S0 = compute_clip_similarity_prompt(prompt, preview_image_path)

        # if prompt is image
        # prompt_image_path = r'C:\Users\Admin\3rd generation\generated_image.png'
        # prompt_image = load_image(prompt_image_path)
        # S0 = compute_clip_similarity(prompt_image, preview_image)  

        Si = [compute_clip_similarity(preview_image, img) for img in rendered_images]

        def resize_image(image, target_size=(256, 256)):
            """Resize an image to the target size."""
            return image.resize(target_size, Image.Resampling.LANCZOS)

        def pil_to_cv(image):
            """Convert PIL Image to OpenCV format and resize."""
            image = resize_image(image)  # Resize image to ensure dimensions match
            return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # Example of checking dimensions
        test_image1 = Image.open(preview_image_path)
        test_image2 = Image.open(rendered_images_paths[0])

        # Assuming preview_image_path is your 'ideal' image and rendered_images_paths contains the rendered variants
        # Ensure you have at least one rendered image path available
        # Preview Image Quality Assessment (Q0)
        # The quality of the preview image is assessed to ensure it meets a minimum quality threshold. This score ranges from 0 to 1.
        if rendered_images_paths:
            print(len(rendered_images_paths))
            Q0 = ssim(pil_to_cv(Image.open(preview_image_path)), pil_to_cv(Image.open(rendered_images_paths[0])), win_size=3)
            Qi = [ssim(pil_to_cv(Image.open(preview_image_path)), pil_to_cv(Image.open(rendered_images_paths[path])), win_size=3) for path in range(1,len(rendered_images_paths))]
        else:
            Q0 = 0  # No comparison available, set to 0 or an appropriate value indicating no data
            Qi = []

        # Geometric Mean of Similarity Scores (Sgeo)
        # To ensure that poor quality on side and back views is penalized more heavily, the geometric mean of the similarity scores is computed.
        # Formula:    S_geo  = 〖( ∏_(i=1 to 12)〖S_i 〗)〗^(1/12)
        # S_geo = np.sum(Si)**(1/len(Si))
        S_geo = np.exp(np.log(Si).mean())

        # Geometric Mean of Quality Scores (Qgeo)
        # The geometric mean of the quality scores is computed to ensure that all rendered images meet a minimum quality threshold.
        # Formula:    Q_geo  = 〖( ∏_(i=1)^12〖Q_i〗)〗^(1/12)
        # print(len(Qi))

        # Q_geo = np.sum(Qi)**(1/len(Qi))
        Q_geo = np.exp(np.log(Qi).mean())
        # print(S0 ,S_geo ,Q0 , Q_geo)

        # Total Similarity Score (Stotal)
        # The total similarity score is the sum of the product of the similarity score between the prompt and preview image (S0) and the geometric mean of the similarity scores (Sgeo), and the product of the quality score of the preview image (Q0) and the geometric mean of the quality scores (Qgeo).
        # Formula:    S_total  = S_0  × S_geo  + Q_0  × Q_geo
        S_total = S0 * S_geo + Q0 * Q_geo

        # print(S_total)

        return {
            "S0": S0,
            "Si": Si,
            "Q0": Q0,
            "Qi": Qi,
            "S_geo": S_geo,
            "Q_geo": Q_geo,
            "S_total": S_total
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Uncomment this section to run the FastAPI app locally
if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=args.port)
