import time
import torch
import clip
from PIL import Image

device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

class ImageModel:
    def __init__(self):
        self.model = None
        
    def compute_clip_similarity(self, img1, img2):
        with torch.no_grad():
            image1_features = self.model.encode_image(img1)
            image2_features = self.model.encode_image(img2)
            similarity = torch.nn.functional.cosine_similarity(image1_features, image2_features).item()
        return similarity
        
    def load_model(self, model: str = "ViT-B/32"):
        try:
            self.model, _ = clip.load(model, device=device)
        except Exception as e:
            print("Loading clip model error")