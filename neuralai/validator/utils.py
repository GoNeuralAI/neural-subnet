import bittensor as bt
import urllib.parse
import aiohttp
import time
import zipfile
import io
import base64

async def validate(val_url: str, prompt: str, uid: int):
    url = urllib.parse.urljoin(val_url, "/validate/")
    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(url, json={"prompt": prompt, "uid": uid}) as response:
                if response.status == 200:
                    results = await response.json()
                    bt.logging.debug(f"===================={uid} : {results}==================")
                    return results
                else:
                    bt.logging.error(f"Generation failed. Please try again.: {response.status}")
        except aiohttp.ClientConnectorError:
            bt.logging.error(f"Failed to connect to the server. Please try to access again: {val_url}.")
        except TimeoutError:
            bt.logging.error(f"The requested time error occured: {val_url}")
        except aiohttp.ClientError as e:
            bt.logging.error(f"Client error occurred: {e} ({val_url})")
        except Exception as e:
            bt.logging.error(f"Error occurred: {e} ({val_url})")
    
    return None