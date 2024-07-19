import bittensor as bt

import time

from neuralai.protocol import NATextSynapse
from neuralai.validator.reward import get_rewards
from neuralai.utils.uids import get_forward_uids
from neuralai.validator.task_manager import TaskManager


def forward(self, synapse: NATextSynapse=None) -> NATextSynapse:
    """
    The forward function is called by the validator every time step.

    It is responsible for querying the network and scoring the responses.

    - Forwarding requests to miners in multiple thread to ensure total time is around 1000 seconds. In each thread, we do:
        - Calculating rewards if needed
        - Updating scores based on rewards
        - Saving the state
    - Normalize weights based on incentive_distribution
    - SET WEIGHTS!
    - Sleep for 300 seconds if needed
    """
    start_time = time.time()
    loop_time = self.config.neuron.task_period
    
    bt.logging.info("Checking available miners...")
    avail_uids = get_forward_uids(self, count=self.config.neuron.challenge_count)
    
    bt.logging.info(f"Selected miners are: {avail_uids}")
    
    forward_uids = self.miner_manager.get_miner_status(uids=avail_uids)
    
    if not forward_uids:
        bt.logging.warning("No miners are available!")
    else:
        bt.logging.info(f"Available miners are: {forward_uids}")
    
    # avail_uids = self.miner_manager.update_miner_status()
    
    # miner_uids = get_selected_uids(self, avails=avail_uids, count=self.config.neuron.challenge_count)

    # bt.logging.info(f'Sending challenges to miners: {miner_uids}')
    
    nas = NATextSynapse()

    if synapse: #in case of Validator API from users
        nas = synapse
        
    else:
        task = self.task_manager.prepare_task()
        nas = NATextSynapse(prompt_text=task, timeout=self.config.generation.timeout)
        
    if task:
        # The dendrite client queries the network.
        if forward_uids:
            bt.logging.info(f"Sending tasks to miners: {task}")
        
        responses = self.dendrite.query(
            # Send the query to selected miner axons in the network.
            axons=[self.metagraph.axons[uid] for uid in forward_uids],
            # Construct a dummy query. This simply contains a single integer.
            synapse=nas,
            timeout=self.config.generation.timeout,
            # All responses have the deserialize function called on them before returning.
            # You are encouraged to define your own deserialization function.
            deserialize=False,
        )
    
        if forward_uids:
            bt.logging.info(f"Received responses from miners: {responses}")
        
        # generation time will be implemented in step 2
        # res_time = [response.dendrite.process_time for response in responses]
        
        # Log the results for monitoring purposes.
        rewards = get_rewards(self, responses=responses, all_uids=avail_uids, for_uids=forward_uids)

        bt.logging.info(f"Updated scores: {rewards}")
        # Update the scores based on the rewards. You may want to define your own update_scores function for custom behavior.
        # self.update_scores(rewards, avail_uids)
    else:
        bt.logging.error(f"No prompt is ready yet")
    # Adjust the scores based on responses from miners.
    
    taken_time = time.time() - start_time
    if taken_time < loop_time:
        bt.logging.info(f"== Taken time: {taken_time} | Sleeping for {loop_time - taken_time} seconds ==")
        time.sleep(loop_time - taken_time)
