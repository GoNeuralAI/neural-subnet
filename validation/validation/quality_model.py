import torch
import torchvision.transforms as transforms
from PIL import Image

device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

class QualityModel:
    def __init__(self):
        self.model = None
        self.preprocess = None
        
    def compute_quality(self, img_path):
        img = Image.open(img_path).convert("RGB")
        img = self.preprocess(img).unsqueeze(0).to(device)
        with torch.no_grad():
            score = self.model(img)
        return score.item()
        
    def load_model(self, model: str = "QualiCLIP"):
        try:
            self.model = torch.hub.load(repo_or_dir="miccunifi/QualiCLIP", source="github", model=model)
            self.model.eval().to(device)
            print(f"device: {device}")
            self.preprocess = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
            ])

        except Exception as e:
            print("Loading QualiClip model error")