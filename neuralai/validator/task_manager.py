import bittensor as bt
import random
from neuralai.utils.taskLib import taskLib

class TaskManager:
    verbose = 1 #0: test | 1: main
    
    def __init__(self):
        super(TaskManager, self).__init__()
                
    async def prepare_task(self, mode = 1):
        #TODO Preparing the input prompting as text or image
        
        prompts = None
        if self.verbose == 1:
            prompts = await self.get_task()
        else:
            prompts = [
                "Bird in flight",
                "Fish swimming in the sea",
                "Butterfly fluttering through the garden",
                "Dog chasing after a ball",
                "Squirrel scurrying up the tree",
                "Hummingbird hovering near the flowers",
                "Frog leaping across the pond",
                "Wind howling through the trees",
                "Waves crashing against the shore",
                "Ant marching across the ground",
                "Rabbit bounding through the meadow",
                "Dragonfly darting between the reeds",
                "Cat prowling along the fence",
                "Dandelion seeds drifting on the breeze",
                "Fireflies glowing in the night",
                "Deer gracefully bounding through the forest",
                "Swarm of bees buzzing around the hive",
                "Dolphin leaping out of the water",
                "Falling leaves drifting to the ground",
                "Tumbleweed rolling across the desert",
                "The car should have a sleek, aerodynamic design."
            ]
            
        task = random.choice(prompts)
        return task
    
    async def get_task(self):
        task = taskLib()
        prompt = await task.get_task()
        return prompt