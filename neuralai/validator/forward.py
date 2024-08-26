import bittensor as bt

import time

from neuralai.protocol import NATextSynapse
from neuralai.validator.reward import (
    get_rewards, calculate_scores
)
from neuralai.utils.uids import get_forward_uids
from neuralai.validator import utils

async def forward(self, synapse: NATextSynapse=None) -> NATextSynapse:
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
    gen_time = loop_time * 4 / 5
    
    bt.logging.info("Checking available miners...")
    avail_uids = get_forward_uids(self, count=self.config.neuron.challenge_count)
    
    bt.logging.info(f"Listed miners: {avail_uids}")
    
    forward_uids = self.miner_manager.get_miner_status(uids=avail_uids)
    
    if not forward_uids:
        bt.logging.warning("No miners available!")
    else:
        bt.logging.info(f"Available miners: {forward_uids}")
    
    nas = NATextSynapse()
    task = None

    if synapse: #in case of Validator API from users
        nas = synapse
    else:
        task = await self.task_manager.prepare_task()
        nas = NATextSynapse(prompt_text=task, timeout=self.config.generation.timeout)
        
    if task:
        # The dendrite client queries the network.
        scores = []
        if forward_uids:
            bt.logging.info(f"Sending tasks to miners: {task}")
        
            responses = self.dendrite.query(
                # Send the query to selected miner axons in the network.
                axons=[self.metagraph.axons[uid] for uid in forward_uids],
                # Construct a dummy query. This simply contains a single integer.
                synapse=nas,
                timeout=gen_time,
                # All responses have the deserialize function called on them before returning.
                # You are encouraged to define your own deserialization function.
                deserialize=False,
            )
            
            for index, response in enumerate(responses):
                try:
                    utils.save_synapse_files(response, forward_uids[index])
                except ValueError as e:
                    print(f"Error saving files for response {forward_uids[index]}: {e}")
                    
                    
            for index, response in enumerate(responses):
                result = await utils.validate(
                    self.config.validation.endpoint, task, int(forward_uids[index])
                )
                scores.append(result)
                
        rewards = get_rewards(responses=scores, all_uids=avail_uids, for_uids=forward_uids)
        
        scores = calculate_scores(rewards)
        bt.logging.info(f"Updated scores: {scores}")
                
        self.update_scores(scores, avail_uids)
    else:
        bt.logging.error(f"No prompt is ready yet")
        
    # Adjust the scores based on responses from miners.
    
    # res_time = [response.dendrite.process_time for response in responses]
    taken_time = time.time() - start_time
    
    if taken_time < loop_time:
        bt.logging.info(f"== Taken time: {taken_time} | Sleeping for {loop_time - taken_time} seconds ==")
        time.sleep(loop_time - taken_time)
