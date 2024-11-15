import os
import time
import torch
import random
import numpy as np
from PIL import Image
from einops import rearrange
from PIL import Image, ImageSequence

from .utils import seed_everything, timing_decorator, auto_amp_inference
from .utils import get_parameter_number, set_parameter_grad_false
from svrm.predictor import MV23DPredictor


class Views2Mesh():
    def __init__(self, mv23d_cfg_path, mv23d_ckt_path, device="cuda:0", use_lite=False):
        '''
            mv23d_cfg_path: config yaml file 
            mv23d_ckt_path: path to ckpt
            use_lite: 
        '''
        self.mv23d_predictor = MV23DPredictor(mv23d_ckt_path, mv23d_cfg_path, device=device)  
        self.mv23d_predictor.model.eval()
        self.order = [0, 1, 2, 3, 4, 5] if use_lite else [0, 2, 4, 5, 3, 1]
        set_parameter_grad_false(self.mv23d_predictor.model)
        print('view2mesh model', get_parameter_number(self.mv23d_predictor.model))

    @torch.no_grad()
    @timing_decorator("views to mesh")
    @auto_amp_inference
    def __call__(
        self,
        views_pil=None, 
        cond_pil=None, 
        gif_pil=None, 
        seed=0, 
        target_face_count = 10000,
        do_texture_mapping = True,
        save_folder='./outputs/test'
    ):
        '''
            can set views_pil, cond_pil simutaously or set gif_pil only
            seed: int
            target_face_count: int 
            save_folder: path to save mesh files
        '''
        save_dir = save_folder
        os.makedirs(save_dir, exist_ok=True)

        if views_pil is not None and cond_pil is not None:
            show_image = rearrange(np.asarray(views_pil, dtype=np.uint8), 
                                   '(n h) (m w) c -> (n m) h w c', n=3, m=2)
            views = [Image.fromarray(show_image[idx]) for idx in self.order] 
            image_list = [cond_pil]+ views
            image_list = [img.convert('RGB') for img in image_list]
        elif gif_pil is not None:
            image_list = [img.convert('RGB') for img in ImageSequence.Iterator(gif_pil)]
        
        image_input = image_list[0]
        image_list = image_list[1:] + image_list[:1]
        
        seed_everything(seed)
        self.mv23d_predictor.predict(
            image_list, 
            save_dir = save_dir, 
            image_input = image_input,
            target_face_count = target_face_count,
            do_texture_mapping = do_texture_mapping
        )
        torch.cuda.empty_cache()
        return save_dir
        