# import torch
# import torchvision.transforms as transforms
# from PIL import Image
# import time
# import numpy as np

# # # Set the device
# # device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

# # # Load the q_model
# # q_model = torch.hub.load(repo_or_dir="miccunifi/QualiCLIP", source="github", model="QualiCLIP")
# # q_model.eval().to(device)

# # # Define the preprocessing pipeline
# # preprocess = transforms.Compose([
# #     transforms.Resize((224, 224)),
# #     transforms.ToTensor(),
# #     transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
# # ])

# # start = time.time()
# # # Load the image
# # img_path = "origin.jpg"
# # img = Image.open(img_path).convert("RGB")

# # # Preprocess the image
# # img = preprocess(img).unsqueeze(0).to(device)
# # img = img.to(device)

# # # Compute the quality score
# # with torch.no_grad(), torch.cuda.amp.autocast():
# #     score = q_model(img)

# # print(f"Image quality score: {score.item()}")

# # img_path = "output_images/image_0.png"
# # img = Image.open(img_path).convert("RGB")

# # # Preprocess the image
# # img = preprocess(img).unsqueeze(0).to(device)

# # # Compute the quality score
# # with torch.no_grad(), torch.cuda.amp.autocast():
# #     score = q_model(img)

# # print(f"Image quality score: {score.item()}")

# # img_path = "origin-2.png"
# # img = Image.open(img_path).convert("RGB")

# # # Preprocess the image
# # img = preprocess(img).unsqueeze(0).to(device)

# # # Compute the quality score
# # with torch.no_grad(), torch.cuda.amp.autocast():
# #     score = q_model(img)

# # print(f"Image quality score: {score.item()}")

# # print(time.time() - start)