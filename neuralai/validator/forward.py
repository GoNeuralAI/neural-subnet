# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# TODO(developer): Set your name
# Copyright © 2023 <your name>

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import bittensor as bt

from neuralai.protocol import NASynapse
from neuralai.validator.reward import get_rewards
from neuralai.utils.uids import get_random_uids
from neuralai.validator.task_manager import TaskManager


def forward(self, synapse: NASynapse=None) -> NASynapse:
    """
    The forward function is called by the validator every time step.

    It is responsible for querying the network and scoring the responses.

    - Forwarding requests to miners in multiple thread to ensure total time is around 1000 seconds. In each thread, we do:
        - Calculating rewards if needed
        - Updating scores based on rewards
        - Saving the state
    - Normalize weights based on incentive_distribution
    - SET WEIGHTS!
    - Sleep for 1000 seconds if needed
    """
    # TODO(developer): Define how the validator selects a miner to query, how often, etc.
    # get_random_uids is an example method, but you can replace it with your own.
    
    bt.logging.info("Checking available miners")
    
    self.miner_manager.update_miner_status()
    
    miner_uids = get_random_uids(self, k=self.config.neuron.sample_size)

    bt.logging.info(f'Sending challenges to miners: {miner_uids}')
    
    nas = NASynapse()

    if synapse: #in case of Validator API from users
        nas = synapse
        
    else:
        task = self.task_manager.prepare_task()
        nas = NASynapse(in_na=task)

    if task:        
        # The dendrite client queries the network.
        responses = self.dendrite.query(
            # Send the query to selected miner axons in the network.
            axons=[self.metagraph.axons[uid] for uid in miner_uids],
            # Construct a dummy query. This simply contains a single integer.
            synapse=nas,
            # All responses have the deserialize function called on them before returning.
            # You are encouraged to define your own deserialization function.
            deserialize=False,
        )
    
        bt.logging.info(f"Received responses from miners: {responses}")
        
        # Log the results for monitoring purposes.
        # rewards = get_rewards(self, query=self.step, responses=responses)

        bt.logging.info(f"Scored responses:")
        # Update the scores based on the rewards. You may want to define your own update_scores function for custom behavior.
        # self.update_scores(rewards, miner_uids)
    else:
        bt.logging.error(f"No prompt is ready yet")
    # TODO(developer): Define how the validator scores responses.
    # Adjust the scores based on responses from miners.
