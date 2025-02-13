from pydantic import BaseModel

class ValidateRequest(BaseModel):
    prompt: str
    uid: int = 0
    verbose: bool = True
    
class ValidateResponse(BaseModel):
    score: float
