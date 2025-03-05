import bittensor as bt
import urllib.parse
import aiohttp
import time
import os
import base64
from dotenv import load_dotenv
from neuralai.miner.s3_bucket import s3_upload, generate_presigned_url
from PIL import Image

load_dotenv()

S3_BUCKET_USE = os.getenv("S3_BUCKET_USE")

def set_status(self, status: str="idle"):
    self.miner_status = status
    
def check_status(self):
    if self.miner_status == "idle":
        return True
    return False
    
def check_validator(self, uid: int, interval: int = 200):
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

def read_file(file_path):
    try:
        mode = 'rb'
        with open(file_path, mode) as file:
            content = file.read()
        return content
    except FileNotFoundError:
        return f"File not found: {file_path}"
    except Exception as e:
        return str(e)

def convert_png_to_jpeg(png_file_path):
    """
    Converts a PNG file to a JPEG file in the same directory with the same base name.
    
    Args:
        png_file_path (str): The full path to the PNG file.
    
    Returns:
        str: The full path to the converted JPEG file.
    """
    try:
        # Ensure the file has a .png extension
        if not png_file_path.lower().endswith('.png'):
            raise ValueError("Provided file is not a PNG file.")
        
        # Generate the JPEG file path
        jpeg_file_path = os.path.splitext(png_file_path)[0] + '.jpeg'
        
        # Open the PNG file and convert it to JPEG
        with Image.open(png_file_path) as img:
            rgb_img = img.convert('RGB')  # Convert to RGB (JPEG does not support transparency)
            rgb_img.save(jpeg_file_path, 'JPEG')
            print(f'Converted: {png_file_path} to {jpeg_file_path}')
        return jpeg_file_path  # Return the path to the new JPEG file
    
    except Exception as e:
        raise RuntimeError(f"Error converting PNG to JPEG: {e}")

async def generate(self, synapse: bt.Synapse) -> bt.Synapse:
    url = urllib.parse.urljoin(self.config.generation.endpoint, "/generate_from_text/")
    timeout = synapse.timeout
    prompt = synapse.prompt_text
    
    extra_prompts = "Angled front view, solid color background, 3d model, high quality"
    enhanced_prompt = f"{prompt}, {extra_prompts}"
    
    if type(synapse).__name__ == "NATextSynapse":
        result = await _generate_from_text(gen_url=url, timeout=timeout, prompt=enhanced_prompt)

        if not result or not result.get('success'):
            bt.logging.warning("Result is None")
            return synapse

        abs_path = os.path.join('generate', result['path'])
        paths = {
            "prev": os.path.join(abs_path, 'mesh.png'),
            "glb": os.path.join(abs_path, 'mesh.glb'),
        }
        
        prev_img_path = convert_png_to_jpeg(paths["prev"])

        try:
            if S3_BUCKET_USE != "TRUE":
                print(paths["prev"])
                synapse.out_prev = base64.b64encode(read_file(prev_img_path)).decode('utf-8')
                synapse.out_glb = base64.b64encode(read_file(paths["glb"])).decode('utf-8')
                synapse.s3_addr = []
            else:
                bt.logging.info("Uploading to S3bucket")
                for key, path in paths.items():
                    file_name = os.path.basename(path)
                    s3_upload(path, f"{self.generation_requests}/{file_name}")
                    synapse.s3_addr.append(generate_presigned_url(f"{self.generation_requests}/{file_name}"))

            bt.logging.info("Valid result")

        except Exception as e:
            bt.logging.error(f"Error reading files: {e}")

    return synapse

async def _generate_from_text(gen_url: str, timeout: int, prompt: str):
    async with aiohttp.ClientSession() as session:
        try:
            bt.logging.debug(f"=================================================")
            client_timeout = aiohttp.ClientTimeout(total=float(timeout))
            
            async with session.post(gen_url, timeout=client_timeout, data={"prompt": prompt}) as response:
                if response.status == 200:
                    result = await response.json()
                    print("Success:", result)
                else:
                    bt.logging.error(f"Generation failed. Please try again.: {response.status}")
                return result
        except aiohttp.ClientConnectorError:
            bt.logging.error(f"Failed to connect to the endpoint. Try to access again: {gen_url}.")
        except TimeoutError:
            bt.logging.error(f"The request to the endpoint timed out: {gen_url}")
        except aiohttp.ClientError as e:
            bt.logging.error(f"An unexpected client error occurred: {e} ({gen_url})")
        except Exception as e:
            bt.logging.error(f"An unexpected error occurred: {e} ({gen_url})")
    
    return None