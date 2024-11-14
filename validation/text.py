import os
import time
import numpy as np
from validation.text_clip_model import TextModel
from pydantic import BaseModel

from rendering import render, load_image

class ValidateRequest(BaseModel):
    prompt: str
    uid: int = 0

DATA_DIR = './results'


class Validation:
    def __init__(self):
        self.text_model = TextModel()
        
        self.init_model()
        
    def validate(self, data: ValidateRequest):
        print("----------------- Validation started -----------------")
        start = time.time()
        prompt = data.prompt
        id = data.uid
        
        prev_img_path = os.path.join(DATA_DIR, f"{data.uid}/preview.png")
        
        S0 = self.text_model.compute_clip_similarity_prompt(prompt, prev_img_path)
        print(f"S0: {S0} - taken time: {time.time() - start}")
        
        return S0
        
    def detect_outliers(self, data, threshold=1.1):
        # Calculate Q1 and Q3
        sorted_data = sorted(data)
        Q1 = np.percentile(sorted_data, 25)
        Q3 = np.percentile(sorted_data, 75)
        
        # Calculate IQR
        IQR = Q3 - Q1
        
        # Determine bounds
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        
        # Identify non-outliers
        non_outliers = [x for x in data if lower_bound <= x <= upper_bound]
        
        return non_outliers
        
    def init_model(self):
        print("loading models")
        """
        
        Loading models needed for text-to-image, image-to-image and image quality models
        After that, calculate the .glb file score
        """
        
        self.text_model.load_model()
        
if __name__ == "__main__":
    prompt = ""
    id = 5
    c = Validation()
    data = ValidateRequest
    data.prompt = prompt
    data.uid = id
    c.validate(data)