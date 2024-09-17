import bittensor as bt
from neuralai.protocol import NAStatus
from typing import List

class MinerManager:
    def __init__(self, validator):
        self.validator = validator
        
    def get_miner_status(self, uids: List[int]):
        all_axons = self.validator.metagraph.axons
        query_axons = [all_axons[uid] for uid in uids]
        synapse = NAStatus(sn_version=self.validator.spec_version)
        responses = self.validator.dendrite.query(
            query_axons,
            synapse,
            deserialize=False,
            timeout=10
        )

        responses = {
            uid: response.status
            for uid, response in zip(uids, responses)
        }
        availables = [uid for uid, status in responses.items() if status == 'idle']
        return availables
