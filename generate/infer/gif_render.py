from svrm.ldm.vis_util import render
from .utils import seed_everything, timing_decorator

class GifRenderer():
    '''
        render frame(s) of mesh using pytorch3d
    '''
    def __init__(self, device="cuda:0"):
        self.device = device

    @timing_decorator("gif render")
    def __call__(
        self, 
        obj_filename, 
        elev=0, 
        azim=0, 
        resolution=512, 
        gif_dst_path='', 
        n_views=120, 
        fps=30, 
        rgb=True
    ):
        render(
            obj_filename,
            elev=elev, 
            azim=azim, 
            resolution=resolution, 
            gif_dst_path=gif_dst_path, 
            n_views=n_views, 
            fps=fps, 
            device=self.device, 
            rgb=rgb
        )
