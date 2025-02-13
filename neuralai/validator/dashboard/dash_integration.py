import json
import os
import time
from hashlib import sha256
from uuid import uuid4
from math import ceil
from substrateinterface import Keypair
from typing import Dict, Any, Optional
import datetime
import asyncio
import requests
from dotenv import load_dotenv
from vectornet.validator.dashboard.model import MinerData, Operation


load_dotenv()

async def generate_header(hotkey: Keypair, body: bytes, signed_for: Optional[str] = None) -> Dict[str, Any]:
    timestamp = round(time.time() * 1000)
    timestampInterval = ceil(timestamp / 1e4) * 1e4
    uuid = str(uuid4())
    headers = {
        "Version": "2",
        "Timestamp": str(timestamp),
        "Uuid": uuid,
        "Signed-By": hotkey.ss58_address,
        "Request-Signature": "0x"
        + hotkey.sign(
            f"{sha256(body).hexdigest()}.{uuid}.{timestamp}.{signed_for or ''}"
        ).hex(),
    }
    if signed_for:
        headers["Signed-For"] = signed_for
        headers["Secret-Signature-0"] = (
            "0x" + hotkey.sign(str(timestampInterval - 1) + "." + signed_for).hex()
        )
        headers["Secret-Signature-1"] = (
            "0x" + hotkey.sign(str(timestampInterval) + "." + signed_for).hex()
        )
        headers["Secret-Signature-2"] = (
            "0x" + hotkey.sign(str(timestampInterval + 1) + "." + signed_for).hex()
        )
    return headers

async def send_data_to_dashboard(miner_data: MinerData, hotkey: Keypair, receiver_hotkey_ss58: str):
    body = json.dumps(miner_data.to_dict()).encode('utf-8')
    print(body)
    headers = await generate_header(hotkey, body, signed_for=receiver_hotkey_ss58)
    
    url = "http://" + os.getenv("DASHBOARD_SERVER_ADDRESS") + os.getenv("ENDPOINT")
    
    
    response = requests.post(url, headers=headers, json=miner_data.to_dict())
    
    if response.status_code == 200:
        print("MinerData sent to dashboard backend successfully")
    else:
        print(f"Failed to send data: {response.status_code} - {response.text}")
        
        
if __name__ == "__main__":
    mock_data = MinerData(
        miner_uid="user_123",
        validator_hotkey="validator_abc",
        prompt="Generate an image of a sunset.",
        final_score=95.5,
        prev_image="image_path/sunset.png",
        glb_file="model_path/sunset.glb",
        response_time=0.256
    )
    asyncio.run(send_data_to_dashboard(mock_data, "5FRyAYjfEREtBU8YXNR3J5qbjqCcSBd4dj8Yr33bHX8ogxfz", "5CaFsXR78pDfrZd7xRPc79tcUFJaM8fDx9MkQ37qhWYuJ7M5"))