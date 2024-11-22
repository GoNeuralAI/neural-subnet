import torch
import open_clip
from PIL import Image

device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

class TextModel:
    def __init__(self):
        self.model = None
        self.preprocess = None
        self.tokenizer = None
        print("init")
        
    def compute_clip_similarity_prompt(self, prompt: str, img_path: str):
        image = Image.open(img_path)  # Change to your image path
        image = self.preprocess(image).unsqueeze(0)
        
        labels_list = [prompt]
        text = self.tokenizer(labels_list)
        
        with torch.no_grad():
            image_features = self.model.encode_image(image)
            text_features = self.model.encode_text(text)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            text_probs = (image_features @ text_features.T)
        return text_probs.item()
        
    def load_model(self, model: str = "ViT-B-32", pretrained: str = "laion2b_s34b_b79k"):
        try:
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(model, pretrained=pretrained)
            self.model.eval()
            self.tokenizer = open_clip.get_tokenizer(model)
        except Exception as e:
            print("Loading open clip model error")