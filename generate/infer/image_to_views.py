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
from mvd.hunyuan3d_mvd_std_pipeline import HunYuan3D_MVD_Std_Pipeline
from mvd.hunyuan3d_mvd_lite_pipeline import Hunyuan3d_MVD_Lite_Pipeline


def save_gif(pils, save_path, df=False):
    # save a list of PIL.Image to gif
    spf = 4000 / len(pils)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    pils[0].save(save_path, format="GIF", save_all=True, append_images=pils[1:], duration=spf, loop=0)
    return save_path
    

class Image2Views():
    def __init__(self, device="cuda:0", use_lite=False):
        self.device = device
        if use_lite:
            self.pipe = Hunyuan3d_MVD_Lite_Pipeline.from_pretrained(
                "./weights/mvd_lite",
                torch_dtype = torch.float16,
                use_safetensors = True,
            )
        else:
            self.pipe = HunYuan3D_MVD_Std_Pipeline.from_pretrained(
                "./weights/mvd_std",
                torch_dtype = torch.float16,
                use_safetensors = True,
            )
        self.pipe = self.pipe.to(device)
        self.order = [0, 1, 2, 3, 4, 5] if use_lite else [0, 2, 4, 5, 3, 1]
        set_parameter_grad_false(self.pipe.unet)
        print('image2views unet model', get_parameter_number(self.pipe.unet))
        
    @torch.no_grad()
    @timing_decorator("image to views")
    @auto_amp_inference
    def __call__(self, pil_img, seed=0, steps=50, guidance_scale=2.0, guidance_curve=lambda t:2.0):
        seed_everything(seed)
        generator = torch.Generator(device=self.device)
        res_img = self.pipe(pil_img, 
                            num_inference_steps=steps,
                            guidance_scale=guidance_scale, 
                            guidance_curve=guidance_curve, 
                            generat=generator).images
        show_image = rearrange(np.asarray(res_img[0], dtype=np.uint8), '(n h) (m w) c -> (n m) h w c', n=3, m=2)
        pils = [res_img[1]]+[Image.fromarray(show_image[idx]) for idx in self.order] 
        torch.cuda.empty_cache()
        return res_img, pils
        