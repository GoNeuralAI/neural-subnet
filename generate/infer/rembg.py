from rembg import remove, new_session
from .utils import timing_decorator

class Removebg():
    def __init__(self, name="u2net"):
        '''
            name: rembg
        '''
        self.session = new_session(name)

    @timing_decorator("remove background")
    def __call__(self, rgb_img, force=False):
        '''
            inputs:
                rgb_img: PIL.Image, with RGB mode expected
                force: bool, input is RGBA mode
            return:
                rgba_img: PIL.Image with RGBA mode
        '''
        if rgb_img.mode == "RGBA":
            if force:
                rgb_img = rgb_img.convert("RGB")
            else:
                return rgb_img
        rgba_img = remove(rgb_img, session=self.session)
        return rgba_img
