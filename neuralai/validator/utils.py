import bittensor as bt
import urllib.parse
import aiohttp
import base64
import os
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

def decode_base64(data, description):
    """Decode base64 data and handle potential errors."""
    if not data:
        raise ValueError(f"{description} data is empty or None.")
    try:
        return base64.b64decode(data)
    except base64.binascii.Error as e:
        raise ValueError(f"Failed to decode {description} data: {e}")

def save_synapse_files(synapse, index, base_dir='validation'):
    # Create a unique subdirectory for each response under validation/results
    save_dir = os.path.join(base_dir, 'results', str(index))
    
    # Ensure the directory exists
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    prev_bytes = decode_base64(synapse.out_prev, "preview")
    texture_bytes = decode_base64(synapse.out_texture, "texture")

    with open(os.path.join(save_dir, 'preview.png'), 'wb') as f:
        f.write(prev_bytes)
    with open(os.path.join(save_dir, 'output.obj'), 'w') as f:
        f.write(synapse.out_obj)
    with open(os.path.join(save_dir, 'output.mtl'), 'w') as f:
        f.write(synapse.out_mtl)
    with open(os.path.join(save_dir, 'output.png'), 'wb') as f:
        f.write(texture_bytes)