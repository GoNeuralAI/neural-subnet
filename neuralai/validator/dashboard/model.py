from pydantic import BaseModel
from datetime import datetime

from datetime import datetime
from pydantic import BaseModel

class MinerData(BaseModel):
    miner_uid: int
    validator_hotkey: str
    prompt: str
    final_score: float
    prev_image: str
    glb_file: str
    response_time: str
    timestamp: datetime  # Defining the datatype for timestamp

    def to_dict(self):
        return {
            "miner_uid": self.miner_uid,
            "validator_hotkey": self.validator_hotkey,
            "prompt": self.prompt,
            "final_score": self.final_score,
            "prev_image": self.prev_image,
            "glb_file": self.glb_file,
            "response_time": self.response_time,
            "timestamp": self.timestamp.isoformat()  # Converting timestamp to ISO format
        }