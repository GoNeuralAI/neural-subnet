import os
import time
import random
import numpy as np
import torch
from torch.cuda.amp import autocast, GradScaler
from functools import wraps

def seed_everything(seed):
    '''
        seed everthing
    '''
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PL_GLOBAL_SEED"] = str(seed)

def timing_decorator(category: str):
    '''
        timing_decorator: record time
    '''
    def decorator(func):
        func.call_count = 0
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs) 
            end_time = time.time()
            elapsed_time = end_time - start_time
            func.call_count += 1
            print(f"[Generation]-[{category}], cost time: {elapsed_time:.4f}s") # huiwen
            return result
        return wrapper
    return decorator

def auto_amp_inference(func):
    '''
        with torch.cuda.amp.autocast()"
            xxx
    '''
    @wraps(func)
    def wrapper(*args, **kwargs):
        with autocast(): 
            output = func(*args, **kwargs) 
        return output
    return wrapper

def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

def set_parameter_grad_false(model):
    for p in model.parameters():
        p.requires_grad = False
