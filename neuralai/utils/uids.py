import random
import bittensor as bt
import numpy as np
from typing import List
import os
import shutil

BASE_DIR = 'validation'
def check_uid_availability(
    metagraph: "bt.metagraph.Metagraph", uid: int, vpermit_tao_limit: int
) -> bool:
    """Check if uid is available. The UID should be available if it is serving and has less than vpermit_tao_limit stake
    Args:
        metagraph (:obj: bt.metagraph.Metagraph): Metagraph object
        uid (int): uid to be checked
        vpermit_tao_limit (int): Validator permit tao limit
    Returns:
        bool: True if uid is available, False otherwise
    """
    # Filter non serving axons.
    if not metagraph.axons[uid].is_serving:
        return False
    # Filter validator permit > 1024 stake.
    if metagraph.validator_permit[uid]:
        if metagraph.S[uid] >= vpermit_tao_limit:
            return False
    # Available otherwise.
    return True

def get_synthetic_forward_uids(
    self, count: int = None, exclude: List[int] = None
) -> np.ndarray:
    
    candidate_uids = []
    avail_uids = []
    
    # bt.logging.debug(f"Uids type: {self.metagraph.n.item()}")

    for uid in range(self.metagraph.n.item()):
        uid_is_available = check_uid_availability(
            self.metagraph, uid, self.config.neuron.vpermit_tao_limit
        )
        uid_is_not_excluded = exclude is None or uid not in exclude

        if uid_is_available:
            avail_uids.append(uid)
            if uid_is_not_excluded:
                candidate_uids.append(uid)
    # If count is larger than the number of available uids, set count to the number of available uids.
    count = min(count, len(avail_uids))
    # Check if candidate_uids contain enough for querying, if not grab all avaliable uids
    available_uids = candidate_uids
    if len(candidate_uids) < count:
        available_uids += random.sample(
            [uid for uid in avail_uids if uid not in candidate_uids],
            count - len(candidate_uids),
        )
    uids = np.array(random.sample(available_uids, count))

    cleanup_results(uids)
    return uids

def get_organic_forward_uids(
    self, count: int = None, exclude: List[int] = None
) -> np.ndarray:
    
    candidate_uids = []
    avail_uids = []
    
    # bt.logging.debug(f"Uids type: {self.metagraph.n.item()}")
    incentives = self.metagraph.I
    miner_info = []
    for uid in range(self.metagraph.n.item()):
        incentive = incentives[uid]
        
        miner_info.append({
            "uid": uid,
            "incentive": incentive,
        })

    sorted_uids = [miner["uid"] for miner in sorted(miner_info, key=lambda x: x["incentive"], reverse=True)]
    print("sorted uids : ********** ", sorted_uids)
    for uid in sorted_uids:
        uid_is_available = check_uid_availability(
            self.metagraph, uid, self.config.neuron.vpermit_tao_limit
        )
        uid_is_not_excluded = exclude is None or uid not in exclude

        if uid_is_available:
            avail_uids.append(uid)
            if uid_is_not_excluded:
                candidate_uids.append(uid)
    # If count is larger than the number of available uids, set count to the number of available uids.
    count = min(count, len(avail_uids))
    # Check if candidate_uids contain enough for querying, if not grab all avaliable uids
    available_uids = candidate_uids
    
    if len(candidate_uids) < count:
        available_uids += random.sample(
            [uid for uid in avail_uids if uid not in candidate_uids],
            count - len(candidate_uids),
        )
    uids = np.array(random.sample(available_uids, count))

    cleanup_results(uids)
    print("final UIDS is *************** this : ", uids)
    uids = [15, 10]
    return uids

def cleanup_results(results):
    for reusult in results:
        reusult_path = os.path.join(BASE_DIR, 'results', str(reusult))
        if os.path.exists(reusult_path) and os.path.isdir(reusult_path):
            shutil.rmtree(reusult_path)