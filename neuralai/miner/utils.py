import bittensor as bt
import urllib.parse
import aiohttp
import time
import zipfile
import io
import base64

def set_status(self, status: str="idle"):
    self.miner_status = status
    
def check_status(self):
    if self.miner_status == "idle":
        return True
    return False
    
def check_validator(self, uid: int, interval: int = 300):
    cur_time = time.time()
    bt.logging.debug(f"Checking validator for UID: {uid} : {self.validators[uid]}")
    
    if uid not in self.validators:
        bt.logging.debug("Adding new validator.")
        self.validators[uid] = {
            "start": cur_time,
            "requests": 1,
        }
    elif cur_time - self.validators[uid]["start"] > interval:
        bt.logging.debug("Resetting validator due to interval.")
        self.validators[uid] = {
            "start": cur_time,
            "requests": 1,
        }
    else:
        bt.logging.debug("Incrementing request count for existing validator.")
        self.validators[uid]["requests"] += 1
        return True
    
    return False

async def generate(self, synapse: bt.Synapse) -> bt.Synapse:
    url = urllib.parse.urljoin(self.config.generation.endpoint, "/generate_from_text/")
    timeout = synapse.timeout
    # bt.logging.debug(f"timeout type: {type(timeout)}")
    prompt = synapse.prompt_text
    synapse_type = type(synapse).__name__
    
    if synapse_type == "NATextSynapse":
        result = await _generate_from_text(gen_url=url, timeout=timeout, prompt=prompt)
        bt.logging.debug(f"Generation result type: {type(result)}")
        
        # Check if the result is None
        if result is None:
            bt.logging.warning("Result is None, returning None")
            
        elif isinstance(result, dict) and all(key in result for key in ["prev", "obj", "mtl", "texture"]):
            synapse.out_prev = result["prev"]
            synapse.out_obj = result["obj"]
            synapse.out_mtl = result["mtl"]
            synapse.out_texture = result["texture"]
            bt.logging.info("Valid result")
        else:
            bt.logging.warning("Result is not valid, returning None")
    
    return synapse

async def _generate_from_text(gen_url: str, timeout: int, prompt: str):
    async with aiohttp.ClientSession() as session:
        try:
            bt.logging.debug(f"=================================================")
            client_timeout = aiohttp.ClientTimeout(total=float(timeout))
            
            async with session.post(gen_url, timeout=client_timeout, data={"prompt": prompt}) as response:
                if response.status == 200:
                    zip_buffer = io.BytesIO(await response.read())
                    with zipfile.ZipFile(zip_buffer, 'r') as zip_file:
                        # Extract each file's data
                        prev_data = zip_file.read('preview.png')
                        obj_data = zip_file.read('output.obj').decode('utf-8')
                        mtl_data = zip_file.read('output.mtl').decode('utf-8')
                        texture_data = zip_file.read('output.png')  # This will be in bytes
                    
                        encoded_prev_data = base64.b64encode(prev_data).decode('utf-8')
                        encoded_texture_data = base64.b64encode(texture_data).decode('utf-8')
                    
                    return {
                        "prev": encoded_prev_data,
                        "obj": obj_data,
                        "mtl": mtl_data,
                        "texture": encoded_texture_data
                    }
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