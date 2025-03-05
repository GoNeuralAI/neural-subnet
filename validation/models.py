from pydantic import BaseModel

class ValidateRequest(BaseModel):
    prompt: str
    uuid: int = 0
    verbose: bool = True
    
class ValidateResponse(BaseModel):
    score: float
