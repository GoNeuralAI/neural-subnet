import bittensor as bt
from neuralai.protocol import NAStatus

class MinerManager:
    def __init__(self, validator):
        self.validator = validator
        
    def get_miner_status(self):
        all_uids = self.validator.metagraph.uids
        synapse = NAStatus()
        responses = self.validator.dendrite.query(
            [self.validator.metagraph.axons[uid] for uid in all_uids],
            synapse,
            deserialize=False,
            timeout=10
        )
        # bt.logging.info(f"Miner Status: {responses}")
        responses = {
            uid: response.status
            for uid, response in zip(all_uids, responses)
        }
        availables = [uid for uid, status in responses.items() if status == 'idle']
        return availables
    
    def update_miner_status(self):
        avail_miners = self.get_miner_status() 
        
        if not avail_miners:
            bt.logging.warning("No miners are available now.")
            
        return avail_miners