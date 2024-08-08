import bittensor as bt

from fastapi import FastAPI, Body
from fastapi.responses import Response
import uvicorn

import base64
from io import BytesIO

import time
import random

app = FastAPI()

@app.post("/generate_from_text/")
async def generate(
    prompt: str = Body()
):
    print(f"prompt: {prompt}")
    timeout, content = await _generate(prompt)
    # buffer = base64.b64encode(buffer.getbuffer()).decode("utf-8")
    print(timeout, content)
    return {
        "timeout": timeout,
        "content": content
    }

#3D Generation
async def _generate(prompt: str):
    start = time.time() #start time
    timeout = random.randint(5, 15)
    # timeout = int(20)
    print(timeout)
    time.sleep(timeout)
    print(f"The generation of a 3D model from text took {time.time() - start} seconds.")
    return timeout, f"successfully generated: {prompt}" 

if __name__ == "__main__":
    uvicorn.run(app=app, host="0.0.0.0", port=8093)