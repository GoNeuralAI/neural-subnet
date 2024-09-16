import bittensor as bt
import urllib.parse
import aiohttp
import base64
import os
import requests
import base64

async def validate(val_url: str, prompt: str, uid: int, timeout: float):
    url = urllib.parse.urljoin(val_url, "/validate/")
    async with aiohttp.ClientSession() as session:
        try:
            client_timeout = aiohttp.ClientTimeout(total=timeout/2)
            async with session.post(url, timeout=client_timeout, json={"prompt": prompt, "uid": uid}) as response:
                if response.status == 200:
                    results = await response.json()
                    bt.logging.debug(f"==================== {uid} : {results} ==================")
                    return results
                else:
                    bt.logging.error(f"Generation failed. Please try again.: {response.status}")
                return {'score': 0}
        except aiohttp.ClientConnectorError:
            bt.logging.error(f"Failed to connect to the server. Please try to access again: {val_url}.")
        except TimeoutError:
            bt.logging.error(f"The requested time error occured: {val_url}")
        except aiohttp.ClientError as e:
            bt.logging.error(f"Client error occurred: {e} ({val_url})")
        except Exception as e:
            bt.logging.error(f"Error occurred: {e} ({val_url})")
    
    return {'score': 0}

def decode_base64(data, description):
    """Decode base64 data and handle potential errors."""
    if not data:
        raise ValueError(f"{description} data is empty or None.")
    try:
        return base64.b64decode(data)
    except base64.binascii.Error as e:
        raise ValueError(f"Failed to decode {description} data: {e}")

file_names = ["preview.png", "output.obj", "output.mtl", "output.png"]

def save_file(file_path, content, is_binary=True):
    mode = 'wb' if is_binary else 'w'
    with open(file_path, mode) as f:
        f.write(content)

def save_synapse_files(synapse, index, base_dir='validation'):
    save_dir = os.path.join(base_dir, 'results', str(index))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if len(synapse.s3_addr) != 0:
        responses = [requests.get(addr) for addr in synapse.s3_addr]
        try:
            for response, file_name in zip(responses, file_names):
                if response.status_code == 200:
                    bt.logging.debug(f"Processing file: {file_name}")
                    save_file(os.path.join(save_dir, file_name), response.content, is_binary=True)
                else:
                    bt.logging.warning(f"Failed to retrieve {file_name}: Status code {response.status_code}")
        except Exception as e:
            bt.logging.error(f"Error occurred while processing files: {e}")
    else:
        try:
            save_file(os.path.join(save_dir, 'preview.png'), decode_base64(synapse.out_prev, "preview"))
            save_file(os.path.join(save_dir, 'output.obj'), synapse.out_obj, False)
            save_file(os.path.join(save_dir, 'output.mtl'), synapse.out_mtl, False)
            save_file(os.path.join(save_dir, 'output.png'), decode_base64(synapse.out_texture, "texture"))
        except Exception as e:
            print(f"Error saving synapse files: {e}")