import os
import time
import numpy as np
from models import ValidateRequest, ValidateResponse
from validation.text_clip_model import TextModel
from validation.image_clip_model import ImageModel
from validation.quality_model import QualityModel

from rendering import render, load_image

DATA_DIR = './results'
EXTRA_PROMPT = 'anime'



class Validation:
    def __init__(self):
        self.text_model = TextModel()
        self.image_model = ImageModel()
        self.quality_model = QualityModel()
        
        self.init_model()
        
    def validate(self, data: ValidateRequest):
        print("----------------- Validation started -----------------")
        start = time.time()
        prompt = data.prompt + " " + EXTRA_PROMPT
        id = data.uid
        
        rendered_images, before_images = render(prompt, id)
        
        prev_img_path = os.path.join(DATA_DIR, f"{data.uid}/preview.png")
        prev_img = load_image(prev_img_path)
        
        Q0 = self.quality_model.compute_quality(prev_img_path)
        print(f"Q0: {Q0}")
        
        S0 = self.text_model.compute_clip_similarity_prompt(prompt, prev_img_path) if Q0 > 0.4 else 0
        print(f"S0: {S0} - taken time: {time.time() - start}")
        if S0 < 0.23:
            return ValidateResponse(score=0)
            
        Ri = self.detect_outliers([self.image_model.compute_clip_similarity(prev_img, img) for img in rendered_images])
        
        Si = self.detect_outliers([self.text_model.compute_clip_similarity_prompt(prompt, before_image) for before_image in before_images])
        
        print(f"R0: taken time: {time.time() - start}")
        
        Qi = self.detect_outliers([self.quality_model.compute_quality(img) for img in before_images])
        
        S_geo = np.exp(np.log(Si).mean())
        R_geo = np.exp(np.log(Ri).mean())
        Q_geo = np.exp(np.log(Qi).mean())
        
        print("---- Rendered images similarities with preview image ---")
        print(Ri)
        print(f"R_geo: {R_geo}")
        
        print("---- Rendered images similarities with text prompt ----")
        print(Si)
        print(f"S_geo: {S_geo}")
        
        print("---- Rendered images quality ----")
        print(Qi)
        print(f"Q_geo: {Q_geo}")
        
        total_score = S0 * 0.2 + S_geo * 0.4 + R_geo * 0.3 + Q_geo * 0.1
        
        print(f"---- Total Score: {total_score} ----")
        
        if total_score < 0.35:
            return ValidateResponse(score=0)
        return ValidateResponse(
            score=total_score
        )
        
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
        self.image_model.load_model()
        self.quality_model.load_model()