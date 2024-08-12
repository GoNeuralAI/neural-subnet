import bittensor as bt
import base64
import os

import time

from neuralai.protocol import NATextSynapse
from neuralai.validator.reward import get_rewards
from neuralai.utils.uids import get_forward_uids
from neuralai.validator.task_manager import TaskManager
from nerualai.validator import utils

def decode_base64(data, description):
    """Decode base64 data and handle potential errors."""
    if not data:
        raise ValueError(f"{description} data is empty or None.")
    try:
        return base64.b64decode(data)
    except base64.binascii.Error as e:
        raise ValueError(f"Failed to decode {description} data: {e}")

async def save_synapse_files(synapse, index, base_dir='validation'):
    # Create a unique subdirectory for each response under validation/results
    save_dir = os.path.join(base_dir, 'results', str(index))
    
    # Ensure the directory exists
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Assuming synapse has these attributes after processing
    prev_data = synapse.out_prev
    obj_data = synapse.out_obj
    mtl_data = synapse.out_mtl
    texture_data = synapse.out_texture

    # Decode the Base64 encoded data with validation
    prev_bytes = decode_base64(prev_data, "preview")
    texture_bytes = decode_base64(texture_data, "texture")

    # Construct file paths
    prev_path = os.path.join(save_dir, 'preview.png')
    obj_path = os.path.join(save_dir, 'output.obj')
    mtl_path = os.path.join(save_dir, 'output.mtl')
    texture_path = os.path.join(save_dir, 'output.png')

    # Save the files
    with open(prev_path, 'wb') as f:
        f.write(prev_bytes)

    with open(obj_path, 'w') as f:
        f.write(obj_data)

    with open(mtl_path, 'w') as f:
        f.write(mtl_data)

    with open(texture_path, 'wb') as f:
        f.write(texture_bytes)

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
        
        for index, response in enumerate(responses):
            try:
                await save_synapse_files(response, forward_uids[index])
            except ValueError as e:
                print(f"Error saving files for response {forward_uids[index]}: {e}")
                
        for index, response in enumerate(responses):
            await utils.validate(self.config.validation.endpoint, response.prompt_text, forward_uids[index])
        
        # if forward_uids:
        #     bt.logging.info(f"Received responses from miners: {responses}")
        
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
    if taken_time < loop_time and forward_uids:
        bt.logging.info(f"== Taken time: {taken_time} | Sleeping for {loop_time - taken_time} seconds ==")
        time.sleep(loop_time - taken_time)
