import bittensor as bt
import urllib.parse
import aiohttp
import base64
import os
import requests
import base64

async def validate(val_url: str, prompt: str, uid: int, timeout: float):
    """
    Validates a request by sending data to the validation endpoint.

    Args:
        val_url (str): Base URL of the validation service.
        prompt (str): The prompt to be validated.
        uid (int): Unique identifier for the request.
        timeout (float): Timeout in seconds for the HTTP request.

    Returns:
        dict: A dictionary containing the validation result or a default response in case of errors.
    """
    url = urllib.parse.urljoin(val_url, "/validate/")
    async with aiohttp.ClientSession() as session:
        try:
            client_timeout = aiohttp.ClientTimeout(total=timeout)

            async with session.post(
                url, timeout=client_timeout, json={"prompt": prompt, "uid": uid}
            ) as response:
                if response.status == 200:
                    results = await response.json()
                    bt.logging.info(f"===== {uid} : {results} =====")
                    return results

                elif response.status == 400:
                    error_message = await response.text()
                    bt.logging.error(f"== {uid} : Bad Request (400). Check input data. Response: {error_message} ==")

                elif response.status == 403:
                    bt.logging.error(f"== {uid} : Forbidden (403). You do not have permission to access this resource. ==")

                elif response.status == 404:
                    bt.logging.error(f"== {uid} : Not Found (404). The requested resource does not exist. ==")

                elif response.status == 408:
                    bt.logging.error(f"== {uid} : Request Timeout (408). The server took too long to respond. ==")

                elif response.status == 500:
                    bt.logging.error(f"== {uid} : Internal Server Error (500). Issue with the server. ==")

                elif response.status == 502:
                    bt.logging.error(f"== {uid} : Bad Gateway (502). Invalid response from an upstream server. ==")

                elif response.status == 503:
                    bt.logging.error(f"== {uid} : Service Unavailable (503). The server is currently unavailable. ==")

                elif response.status == 504:
                    bt.logging.error(f"== {uid} : Gateway Timeout (504). No timely response from upstream server. ==")

                else:
                    error_message = await response.text()
                    bt.logging.error(f"== {uid} : Unexpected Response ({response.status}). Response: {error_message} ==")

                return {'score': 0}

        except aiohttp.ClientConnectorError:
            bt.logging.error(f"Failed to connect to the server. Please check the URL: {val_url}.")

        except aiohttp.ServerTimeoutError:
            bt.logging.error(f"Request timed out. The server did not respond in time: {val_url}.")

        except TimeoutError:
            bt.logging.error(f"Timeout error occurred while trying to reach: {val_url}.")

        except aiohttp.ClientError as e:
            bt.logging.error(f"Client error occurred: {e} ({val_url})")

        except Exception as e:
            bt.logging.error(f"Critical error occurred: {e} ({val_url})")

        return {'score': 0}

def decode_base64(data, description):
    """Decode base64 data and handle potential errors."""
    if not data:
        raise ValueError(f"{description} data is empty or None.")
    try:
        return base64.b64decode(data)
    except base64.binascii.Error as e:
        raise ValueError(f"Failed to decode {description} data: {e}")

file_names = ["preview.png", "output.glb"]

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
            save_file(os.path.join(save_dir, 'output.glb'), decode_base64(synapse.out_glb, "glb"))
        except Exception as e:
            print(f"Error saving synapse files: {e}")