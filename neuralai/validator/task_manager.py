import bittensor as bt

class TaskManager:
    verbose = 0 #0: test | 2: main
    
    def __init__(self):
        super(TaskManager, self).__init__()
        
    def prepare_task(self, mode = 1):
        #TODO Preparing the input prompting as text or image
        
        if mode == 1:
            return "red apple"
        else:
            return "image"
        return None