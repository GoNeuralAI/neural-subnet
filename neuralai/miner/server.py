import bittensor as bt

from fastapi import FastAPI
from fastapi.responses import Response

import base64
from io import BytesIO

import time
import random

app = FastAPI()

@app.post("/generate/")
async def generate(
    prompt: str
):
    buffer = await _generate(prompt)
    buffer = base64.b64encode(buffer.getbuffer()).decode("utf-8")
    return Response(content=buffer, media_type="application/octet-stream")

#3D Generation
async def _generate(prompt: str) -> BytesIO:
    start = time() #start time
    timeout = random.randint(5, 15)
    time.sleep(timeout)
    bt.logging.info(f"The generation of a 3D model from text took {time() - start} seconds.")
    return None