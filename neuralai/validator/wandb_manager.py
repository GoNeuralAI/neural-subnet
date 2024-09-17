import bittensor as bt
import datetime
import wandb
from dotenv import load_dotenv

from neuralai import __version__ as version

load_dotenv()

class WandbManager:
    def __init__(self, validator=None):
        self.wandb = None
        self.wandb_start = datetime.date.today()
        self.validator = validator
        
        if not self.validator.config.wandb.off:
            self.init_wandb()
        else:
            bt.logging.warning(
                "Running validators without Wandb. Recommend to add Wandb!"
            )
            
            
    def init_wandb(self):
        bt.logging.debug("Init wandb")
        
        """Creates a new wandb for validators' lgos"""
        
        self.wandb_start = datetime.date.today()
        current = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        
        name = f"validator-{self.validator.uid}--{version}--{current}"
        self.wandb = wandb.init(
            anonymous="allow",
            name=name,
            project="great",
            entity="creativity111",
            config={
                "uid":self.validator.uid,
                "hotkey":self.validator.wallet.hotkey.ss58_address,
                "version":version,
                "type":"validator",
            }
        )
        
        bt.logging.debug(f"Init a new Wandb: {name}")