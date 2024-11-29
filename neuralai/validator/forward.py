import time
import asyncio
import datetime
import numpy as np
import bittensor as bt

from neuralai.protocol import NATextSynapse
from neuralai.validator.reward import (
    get_rewards, normalize
)
from neuralai.utils.uids import get_forward_uids
from neuralai.validator import utils
from neuralai import __version__ as version


async def handle_response(response, uid, config, nas_prompt_text):
    try:
        utils.save_synapse_files(response, uid)
    except ValueError as e:
        print(f"Error saving files for response {uid}: {e}")

    result = await utils.validate(config.validation.endpoint, nas_prompt_text, int(uid), timeout=config.neuron.task_period / 3 * 2)
    process_time = response.dendrite.process_time
    return result['score'], (process_time if process_time and process_time > 10 else 0)


async def forward(self, synapse: NATextSynapse=None) -> NATextSynapse:
    """
    The forward function is called by the validator every time step.

    It is responsible for querying the network and scoring the responses.

    - Forwarding requests to miners in multiple thread to ensure total time is around 300 seconds. In each thread, we do:
        - Calculating rewards if needed
        - Updating scores based on rewards
        - Saving the state
    - Normalize weights based on incentive_distribution
    - SET WEIGHTS!
    - Sleep for 300 seconds if needed
    """
    start_time = time.time()
    loop_time = self.config.neuron.task_period
    val_scores = []
    
    # wandb
    if not self.config.wandb.off:
        today = datetime.date.today()
        if self.wandb_manager.wandb_start != today:
            self.wandb_manager.wandb.finish()
            self.wandb_manager.init_wandb()
    
    bt.logging.info('=' * 60)
    bt.logging.inf(f"New Epoch v{version}")
    bt.logging.info("Checking Available Miners.....")
    avail_uids = get_forward_uids(self, count=self.config.neuron.challenge_count)
    
    bt.logging.info(f"Listed Miners Are: {avail_uids}")
    
    forward_uids = self.miner_manager.get_miner_status(uids=avail_uids)
    
    if len(forward_uids) == 0:
        bt.logging.warning("No Miners Are Available!")
        val_scores = [0 for _ in avail_uids] 
        self.update_scores(val_scores, avail_uids)
    else:
        bt.logging.info(f"Available Miners: {forward_uids}")
    
        nas = NATextSynapse()
        task = None

        if synapse: #in case of Validator API from users
            nas = synapse
        else:
            task = await self.task_manager.prepare_task()
            nas = NATextSynapse(prompt_text=task, timeout=loop_time / 3)
            
        if nas.prompt_text:
            # The dendrite client queries the network.
            process_time = []
            time_rate = self.config.validator.time_rate
            bt.logging.info(f"======== Currnet Task Prompt: {task} ========")
        
            responses = self.dendrite.query(
                # Send the query to selected miner axons in the network.
                axons=[self.metagraph.axons[uid] for uid in forward_uids],
                # Construct a dummy query. This simply contains a single integer.
                synapse=nas,
                timeout=loop_time / 3,
                # All responses have the deserialize function called on them before returning.
                # You are encouraged to define your own deserialization function.
                deserialize=False,
            )
            bt.logging.info(f"Responses Received")
            
            start_vali_time = time.time()
            
            tasks = [
                handle_response(response, forward_uids[index], self.config, nas.prompt_text)
                for index, response in enumerate(responses)
            ]
            results = await asyncio.gather(*tasks)
            
            val_scores, process_time = zip(*results)

            # Update rewards and scores
            scores = get_rewards(val_scores, avail_uids, forward_uids)
            f_val_scores = normalize(scores)
            bt.logging.info('-' * 40)
            bt.logging.info("=== 3D Object Validation Scores ===", np.round(scores, 3))
            scores = get_rewards(process_time, avail_uids, forward_uids)
            f_time_scores = normalize(scores)
            bt.logging.info("=== Generation Time Scores ===", np.round(scores, 3))
            
            # Considered with the generation time score 0.1
            final_scores = [s * (1 - time_rate) + time_rate * (1 - t) if t else s for t, s in zip(f_time_scores, f_val_scores)]
            
            bt.logging.info("=== Total Scores ===", np.round(final_scores, 3))
            bt.logging.info('-' * 40)
            
            self.update_scores(final_scores, avail_uids)

            bt.logging.info(f"Scoring Taken Time: {time.time() - start_vali_time:.1f}s")
        else:
            bt.logging.error(f"No prompt is ready yet")
        
    # Adjust the scores based on responses from miners.
    
    # res_time = [response.dendrite.process_time for response in responses]
    taken_time = time.time() - start_time
    
    if taken_time < loop_time:
        bt.logging.info(f"== Taken Time: {taken_time:.1f}s | Sleeping For {loop_time - taken_time:.1f}s ==")
        bt.logging.info('=' * 60)
        time.sleep(loop_time - taken_time)
