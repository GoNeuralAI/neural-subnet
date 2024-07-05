import bittensor as bt
import uvicorn

from fastapi import FastAPI, Body
from fastapi.responses import Response

import base64
from io import BytesIO

import time
import random

app = FastAPI()

@app.post("/generate/")
async def generate(
    prompt: str = Body()
):
    buffer = await _generate(prompt)
    # buffer = base64.b64encode(buffer.getbuffer()).decode("utf-8")
    return Response(content=buffer)

#3D Generation
async def _generate(prompt: str) -> BytesIO:
    start = time.time() #start time
    timeout = random.randint(5, 15)
    time.sleep(timeout)
    bt.logging.info(f"The generation of a 3D model from text took {time.time() - start} seconds.")
    return None

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port="8093")
