from pydantic import BaseModel

class ValidateRequest(BaseModel):
    prompt: str
    uid: int = 0
    
class ValidateResponse(BaseModel):
    score: float
