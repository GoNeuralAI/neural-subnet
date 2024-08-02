import bittensor as bt
import random

class TaskManager:
    verbose = 0 #0: test | 1: main
    
    def __init__(self):
        super(TaskManager, self).__init__()
        
    def prepare_task(self, mode = 1):
        #TODO Preparing the input prompting as text or image
        
        task = None
        if self.verbose == 1:
            if mode == 1:
                task = "red apple on the desk"
            else:
                return None # Image
        else:
            examples = [
                # "Bird in flight",
                # "Fish swimming in the sea",
                # "Butterfly fluttering through the garden",
                # "Dog chasing after a ball",
                # "Squirrel scurrying up the tree",
                # "Hummingbird hovering near the flowers"
                # "Frog leaping across the pond",
                # "Wind howling through the trees",
                # "Waves crashing against the shore",
                # "Ant marching across the ground",
                # "Rabbit bounding through the meadow",
                # "Dragonfly darting between the reeds",
                # "Cat prowling along the fence",
                # "Dandelion seeds drifting on the breeze",
                # "Fireflies glowing in the night",
                # "Deer gracefully bounding through the forest",
                # "Swarm of bees buzzing around the hive",
                # "Dolphin leaping out of the water",
                # "Falling leaves drifting to the ground",
                # "Tumbleweed rolling across the desert",
                "The car should have a sleek, aerodynamic design."
            ]
            task = random.choice(examples)
            
        return task