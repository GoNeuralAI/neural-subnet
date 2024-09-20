import time
from typing import Tuple
# Bittensor
import bittensor as bt

# import base validator class which takes care of most of the boilerplate
from neuralai.base.validator import BaseValidatorNeuron
# Bittensor Validator Template:
from neuralai.validator import forward
from neuralai.protocol import NATextSynapse
from neuralai.validator.task_manager import TaskManager
from neuralai.validator.miner_manager import MinerManager
from neuralai.validator.wandb_manager import WandbManager


class Validator(BaseValidatorNeuron):
    """
    Your validator neuron class. You should use this class to define your validator's behavior. In particular, you should replace the forward function with your own logic.

    This class inherits from the BaseValidatorNeuron class, which in turn inherits from BaseNeuron. The BaseNeuron class takes care of routine tasks such as setting up wallet, subtensor, metagraph, logging directory, parsing config, etc. You can override any of the methods in BaseNeuron if you need to customize the behavior.

    This class provides reasonable default behavior for a validator such as keeping a moving average of the scores of the miners and using them to set weights at the end of each epoch. Additionally, the scores are reset for new hotkeys at the end of each epoch.
    """
    
    def __init__(self, config=None):
        super(Validator, self).__init__(config=config)
        
        bt.logging.info("load_state()")
        self.load_state()
        self.task_manager = TaskManager()
        self.miner_manager = MinerManager(validator=self)
        self.wandb_manager = WandbManager(validator=self)

        # TODO(developer): Anything specific to your use case you can do here

    async def forward(self, synapse: NATextSynapse=None):
        """
        Validator forward pass. Consists of:
        - Generating the query
        - Querying the miners
        - Getting the responses
        - Rewarding the miners
        - Updating the scores
        """
        return await forward(self, synapse)
    
    async def forward_fn(self, synapse: NATextSynapse=None):
        time.sleep(5)
        return await self.forward(synapse)
    
    async def blacklist_fn(self, synapse: NATextSynapse) -> Tuple[bool, str]:
        # TODO add hotkeys to blacklist here as needed
        # blacklist the hotkeys mining on the subnet to prevent any potential issues
        #hotkeys_to_blacklist = [h for i,h in enumerate(self.hotkeys) if self.metagraph.S[i] < 20000 and h != self.wallet.hotkey.ss58_address]
        #if synapse.dendrite.hotkey in hotkeys_to_blacklist:
        #    return True, "Blacklisted hotkey - miners can't connect, use a diff hotkey."
        return False, ""

    async def priority_fn(self, synapse: NATextSynapse) -> float:
        # high priority for organic traffic
        return 1000000.0

# The main function parses the configuration and runs the validator.
if __name__ == "__main__":
    with Validator() as validator:
        while True:
            bt.logging.info(f"Validator running... {time.time()}")
            time.sleep(100)
