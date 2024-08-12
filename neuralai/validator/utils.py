import bittensor as bt
import urllib.parse
import aiohttp
import time
import zipfile
import io
import base64

async def validate(val_url: str, prompt: str, uid: int):
    async with aiohttp.ClientSession() as session:
        try:
            bt.logging.debug(f"=================================================")
            async with session.post(val_url, data={"prompt": prompt, "uid": uid}) as response:
                if response.status == 200:
                    results = await response.json()
                else:
                    bt.logging.error(f"Generation failed. Please try again.: {response.status}")
        except aiohttp.ClientConnectorError:
            bt.logging.error(f"Failed to connect to the endpoint. Try to access again: {gen_url}.")
        except TimeoutError:
            bt.logging.error(f"The request to the endpoint timed out: {gen_url}")
        except aiohttp.ClientError as e:
            bt.logging.error(f"An unexpected client error occurred: {e} ({gen_url})")
        except Exception as e:
            bt.logging.error(f"An unexpected error occurred: {e} ({gen_url})")
    
    return None