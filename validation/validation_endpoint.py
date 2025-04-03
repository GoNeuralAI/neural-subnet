import os
import time
import numpy as np
from fastapi import HTTPException
from models import ValidateRequest, ValidateResponse
from validation.text_clip_model import TextModel
from validation.image_clip_model import ImageModel
from validation.quality_model import QualityModel
from validation.text_similarity_model import TextSimilarityModel
from claude_integration import get_render_img_descs, get_prev_img_desc
from rendering import render, load_image
from image_insight import ImageAnalysisToolkit

DATA_DIR = './validation/results'
EXTRA_PROMPT = 'anime'


class Validation:
    def __init__(self):
        self.text_model = TextModel()
        self.image_model = ImageModel()
        self.quality_model = QualityModel()
        self.text_similarity_model = TextSimilarityModel()
        self.image_vision_model = ImageAnalysisToolkit()
        
        self.init_model()
        
    def validate(self, data: ValidateRequest):
        try:
            print("----------------- Validation started -----------------")
            start = time.time()
            prompt = data.prompt
            id = data.uuid
            print(prompt, id)
            print("Rendering 3D mesh file.....")    
            rendered_images, before_images, render_image_paths = render(prompt, id)
            
            image_descs = get_render_img_descs()
            print(image_descs)
            render_vectors = self.text_similarity_model.fetch_vectors(image_descs)
            
            prompt_vector = self.text_similarity_model.fetch_vectors([prompt])[0]
            
            prev_img_path = os.path.join(DATA_DIR, f"{data.uuid}/preview.jpeg")

            image_paths = render_image_paths + [prev_img_path]

            is_like_real_object_image = self.image_vision_model.analyze_images(image_paths)

            print(f"Is real object image: {is_like_real_object_image}")

            if is_like_real_object_image == False:
                return ValidateResponse(score=0)

            print("Loading preview image.....")
            prev_img = load_image(prev_img_path)
            prev_img_desc = get_prev_img_desc(prev_img_path)
            prev_img_vector = self.text_similarity_model.fetch_vectors([prev_img_desc])
            
            Q0 = self.quality_model.compute_quality(prev_img_path)
            print(f"Q0: {Q0}")
            
            S0 = self.text_similarity_model.compute_semantic_similarity(prompt_vector, prev_img_vector)[0] if Q0 > 0.15 else 0
            print(f"S0: {S0} - taken time: {time.time() - start}")
            if S0 < 0.23:
                return ValidateResponse(score=0)
                
            Ri = self.detect_outliers([self.image_model.compute_clip_similarity(prev_img, img) for img in rendered_images])

            Si = self.detect_outliers(self.text_similarity_model.compute_semantic_similarity(prompt_vector, render_vectors))
            
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
            
            total_score = S0 * 0.25 + S_geo * 0.5 + R_geo * 0.3 + Q_geo * 0.1
            
            print(f"---- Total Score: {total_score} ----")
            
            if total_score < 0.35:
                return ValidateResponse(score=0)
            return ValidateResponse(score=total_score)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
        
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
        """
        loading scoring models
        """

        print("loading models")
        
        self.text_model.load_model()
        self.image_model.load_model()
        self.quality_model.load_model()
        self.text_similarity_model.load_model()
