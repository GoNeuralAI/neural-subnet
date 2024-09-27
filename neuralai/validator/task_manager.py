import bittensor as bt
import random
from neuralai.utils.taskLib import taskLib

class TaskManager:
    verbose = True  # False : test | True: main
    
    def __init__(self):
        super(TaskManager, self).__init__()
                
    async def prepare_task(self, mode = 1):
        #TODO Preparing the input prompting as text or image
        
        prompts = None
        if self.verbose == True:
            prompts = await self.get_task()
        else:
            prompts = [
                "Three-quarter view Kettle",
                "Shark with teeth",
                "Toilet",
                "Soccer Ball",
                "donut with icing",
                "moon",
                "red diamond",
                "Diagonal view Chair",
                "Washing machine",
                "magic car"
            ]
            
        task = random.choice(prompts)
        return task
    
    async def get_task(self):
        task = taskLib()
        prompt = await task.get_task()
        return prompt