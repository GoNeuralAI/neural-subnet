import time
import asyncio
import typing
from typing import Tuple
import bittensor as bt

# import base miner class which takes care of most of the boilerplate
from neuralai.base.miner import BaseMinerNeuron
from neuralai.protocol import NATextSynapse, NAImageSynapse, NAStatus
from neuralai.miner.utils import set_status, check_status, generate, check_validator

class Miner(BaseMinerNeuron):
    
    def __init__(self, config=None):
        super(Miner, self).__init__(config=config)

        # TODO(developer): Anything specific to your use case you can do here

        self.validators = {}
        self.generation_requests = 0
        
        set_status(self, self.config.miner.status)

    async def forward_text(
        self, synapse: NATextSynapse
    ) -> NATextSynapse:
        # TODO(developer): Replace with actual implementation logic.
        uid = self.metagraph.hotkeys.index(synapse.dendrite.hotkey)
        
        if not check_status(self):
            bt.logging.warning("Couldn't perform the Generation right now.")
            return synapse
        self.generation_requests += 1
        
        bt.logging.info(f"====== Received a task. Validator uid : {uid}, hotkey : {synapse.dendrite.hotkey} ======")
        bt.logging.info(f"== {synapse.prompt_text} ==")
        
        set_status(self, "generation")
        # Send gpu id as a parameter for multi gpu
        start = time.time()
        synapse = await generate(self, synapse)
        
        self.generation_requests -= 1
        if self.generation_requests < self.config.miner.concurrent_limit:
            set_status(self)
            
        bt.logging.info(f"====== 3D Generation Ended : Taken Time {time.time() - start:.1f}s ======")
        
        return synapse
    
    async def forward_image(
        self, synapse: NAImageSynapse
    ) -> NAImageSynapse:
        """
        For the synapse from the end users to validators
        
        Processes the incoming 'NAImageSynapse' synapse by performing a predefined operation on the input data.
        This method should be replaced with actual logic relevant to the miner's purpose.

        Args:
            synapse (neuralai.protocol.NAImageSynapse): The synapse object containing the 'NAImageSynapse_input' data.

        Returns:
            neuralai.protocol.NAImageSynapse: The synapse object with the 'NAImageSynapse_output' field set to twice the 'NATextSynapse_input' value.

        The 'forward' function is a placeholder and should be overridden with logic that is appropriate for
        the miner's intended operation. This method demonstrates a basic transformation of input data.
        """
        # TODO(developer): Replace with actual implementation logic.
        return synapse

    async def blacklist(self, synapse: NATextSynapse) -> Tuple[bool, str]:
        try:
            if synapse.dendrite is None or synapse.dendrite.hotkey is None:
                bt.logging.warning("Received a request without a dendrite or hotkey.")
                return True, "Missing dendrite or hotkey"

            uid = self.metagraph.hotkeys.index(synapse.dendrite.hotkey)
            if (
                not self.config.blacklist.allow_non_registered
                and synapse.dendrite.hotkey not in self.metagraph.hotkeys
            ):
                # Ignore requests from un-registered entities.
                bt.logging.warning(
                    f"Blacklisting un-registered hotkey {synapse.dendrite.hotkey}"
                )
                return True, "Unrecognized hotkey"

            if self.config.blacklist.force_validator_permit:
                # If the config is set to force validator permit, then we should only allow requests from validators.
                if not self.metagraph.validator_permit[uid]:
                    bt.logging.warning(
                        f"Blacklisting a request from non-validator hotkey {synapse.dendrite.hotkey}"
                    )
                    return True, "Non-validator hotkey"

            # if check_validator(self, uid=uid, interval=int(self.config.miner.gen_interval)):
            #     bt.logging.warning(
            #         f"Too many requests from {synapse.dendrite.hotkey}"
            #     )
            #     return True, "Non-validator hotkey"

            bt.logging.trace(
                f"Not Blacklisting recognized hotkey {synapse.dendrite.hotkey}"
            )
            return False, "All passed!"
        except Exception as e:
            return False, "Hotkey recognized!"
    
    async def blacklist_text(self, synapse: NATextSynapse) -> Tuple[bool, str]:
        return await self.blacklist(synapse)
    
    async def blacklist_image(self, synapse: NAImageSynapse) -> Tuple[bool, str]:
        return await self.blacklist(synapse)

    async def forward_status(self, synapse: NAStatus) -> NAStatus:
        bt.logging.info(f"Current Miner Status: {self.miner_status}, {self.generation_requests}")
        synapse.status = self.miner_status
        if synapse.sn_version > self.spec_version:
            bt.logging.warning(
                "Current subnet version is older than validator subnet version. Please update the miner!"
            )
        elif synapse.sn_version < self.spec_version:
            bt.logging.warning(
                "Current subnet version is higher than validator subnet version. You can ignore this warning!"
            )
            
        if self.generation_requests >= self.config.miner.concurrent_limit:
            set_status(self, "generation")
            
        return synapse
    
    async def blacklist_status(self, synapse: NAStatus) -> Tuple[bool, str]:
        return False, "All passed!"
    
    async def priority(self, synapse: NATextSynapse) -> float:
        """
        The priority function determines the order in which requests are handled. More valuable or higher-priority
        requests are processed before others. You should design your own priority mechanism with care.

        This implementation assigns priority to incoming requests based on the calling entity's stake in the metagraph.

        Args:
            synapse (template.protocol.NATextSynapse): The synapse object that contains metadata about the incoming request.

        Returns:
            float: A priority score derived from the stake of the calling entity.

        Miners may receive messages from multiple entities at once. This function determines which request should be
        processed first. Higher values indicate that the request should be processed first. Lower values indicate
        that the request should be processed later.

        Example priority logic:
        - A higher stake results in a higher priority value.
        """
        if synapse.dendrite is None or synapse.dendrite.hotkey is None:
            bt.logging.warning("Received a request without a dendrite or hotkey.")
            return 0.0
        
        # TODO(developer): Define how miners should prioritize requests.
        caller_uid = self.metagraph.hotkeys.index(
            synapse.dendrite.hotkey
        )  # Get the caller index.
        priority = float(
            self.metagraph.S[caller_uid]
        )  # Return the stake as the priority.
        bt.logging.trace(
            f"Prioritizing {synapse.dendrite.hotkey} with value: {priority}"
        )
        return priority

# This is the main function, which runs the miner.
if __name__ == "__main__":
    with Miner() as miner:
        while True:
            bt.logging.info(f"Miner running... {time.time()}")
            time.sleep(600)
