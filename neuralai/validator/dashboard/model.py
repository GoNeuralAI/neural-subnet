from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

class Operation(BaseModel):
    request_type: str
    s_f: str
    score: float
    timestamp: str

class MinerData(BaseModel):
    miner_uid: int
    total_storage_size: float
    operations: List[Operation]
    request_cycle_score: float
    weight: float
    passed_request_count: int

    def to_dict(self):
        return {
            "miner_uid": self.miner_uid,
            "total_storage_size": self.total_storage_size,
            "operations": [op.__dict__ for op in self.operations],  # Serialize each Operation
            "request_cycle_score": self.request_cycle_score,
            "weight": self.weight,
            "passed_request_count": self.passed_request_count,
        }