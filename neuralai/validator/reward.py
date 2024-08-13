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
import numpy as np
from typing import List
import bittensor as bt
from neuralai.protocol import NATextSynapse

def get_rewards(responses: List,  all_uids: List,  for_uids: List) -> np.ndarray:
    # Get all the reward results by iteratively calling your reward() function.
    # Cast response to int as the reward function expects an int type for response.
    
    # Remove any None values
    # responses = [response for response in responses if response.out_obj is not "obj"]
    return np.array(
        [responses[for_uids.index(uid)]['score'] if uid in for_uids else 0 for uid in all_uids]
    )

def calculate_scores(rewards):
    """
    Normalize the rewards to a range of [0, 1] and apply the transformation y = x^2.
    
    Args:
        rewards (list): A list of reward values.
    
    Returns:
        list: A list of transformed scores.
    """
    # Find max and min values
    max_reward = max(rewards)
    min_reward = min(rewards)

    # Normalize the rewards to [0, 1]
    normalized_rewards = [
        (r - min_reward) / (max_reward - min_reward) if max_reward > min_reward else 0
        for r in rewards
    ]

    # Apply the transformation y = x^2
    scores = [x**2 for x in normalized_rewards]

    return scores
