import bittensor as bt
import urllib.parse
import aiohttp
from aiohttp import ClientTimeout

def set_status(self, status: str="idle"):
    self.miner_status = status

async def generate(self, synapse: bt.Synapse) -> bt.Synapse:
    url = urllib.parse.urljoin(self.config.generation.endpoint, "/generate_from_text/")
    timeout = synapse.timeout
    # bt.logging.debug(f"timeout type: {type(timeout)}")
    prompt = synapse.prompt_text
    
    result = await _generate_from_text(gen_url=url, timeout=timeout, prompt=prompt)
    bt.logging.debug(f"generation result: {type(result)}")
    
    synapse.out_obj = result
    
    return synapse

async def _generate_from_text(gen_url: str, timeout: int, prompt: str):
    client_timeout = ClientTimeout(total=float(timeout))
    async with aiohttp.ClientSession() as session:
        try:
            bt.logging.debug(f"=================================================")
            
            async with session.post(gen_url, data={"prompt": prompt}) as response:
                if response.status == 200:
                    result = await response.text()
                    bt.logging.info(f"Generated successfully: Size = {len(result)}")
                    return result
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