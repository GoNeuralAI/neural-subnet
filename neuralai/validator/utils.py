import bittensor as bt
import urllib.parse
import aiohttp
import base64
import os
import requests
import base64
from PIL import Image
import io

def detect_image_type(base64_string):
    """
    Detect whether a base64-encoded image is PNG or JPEG.

    Args:
        base64_string (str): The base64-encoded image data.

    Returns:
        str: 'png', 'jpeg', or 'unknown' depending on the file type.
    """
    try:
        # Decode the base64 string into bytes
        image_bytes = base64.b64decode(base64_string)

        # Check for PNG magic number
        if image_bytes[:8] == b'\x89PNG\r\n\x1a\n':
            return 'png'

        # Check for JPEG magic number
        if image_bytes[:3] == b'\xFF\xD8\xFF':
            return 'jpeg'

        # If no match, return unknown
        return 'unknown'

    except Exception as e:
        raise RuntimeError(f"Error detecting image type: {e}")


def convert_to_jpeg(image):
    """
    Convert an image to JPEG format.

    Args:
        image (PIL.Image.Image): The input image.

    Returns:
        PIL.Image.Image: The converted image in JPEG format.
    """
    try:
        # Convert the image to RGB (JPEG does not support transparency)
        return image.convert('RGB')
    except Exception as e:
        raise RuntimeError(f"Error converting image to JPEG: {e}")


def resize_image(image, size=(200, 200)):
    """
    Resize an image to the specified dimensions.

    Args:
        image (PIL.Image.Image): The input image.
        size (tuple): Desired image size, default is (200, 200).

    Returns:
        PIL.Image.Image: The resized image.
    """
    try:
        return image.resize(size, Image.Resampling.LANCZOS)
    except Exception as e:
        raise RuntimeError(f"Error resizing image: {e}")


def process_base64_image(base64_string, size=(200, 200)):
    """
    Process a base64-encoded image: check format, convert to JPEG if needed,
    resize it to the specified dimensions, and return the final image as a
    base64-encoded string.

    Args:
        base64_string (str): The base64-encoded image data.
        size (tuple): Desired image size, default is (200, 200).

    Returns:
        str: Base64-encoded string of the final processed image in JPEG format.
    """
    try:
        # Step 1: Detect the image type
        image_type = detect_image_type(base64_string)
        if image_type == 'unknown':
            raise ValueError("Unsupported image format or corrupted base64 string.")

        # Step 2: Decode the base64 string into an image
        image_data = base64.b64decode(base64_string)
        image = Image.open(io.BytesIO(image_data))

        # Step 3: If the image is not JPEG, convert it to JPEG
        if image_type != 'jpeg':
            bt.logging.debug("Image is not JPEG. Converting to JPEG...")
            image = convert_to_jpeg(image)

        # Step 4: Resize the image
        resized_image = resize_image(image, size)

        # Step 5: Save the resized image as a JPEG into a buffer
        buffer = io.BytesIO()
        resized_image.save(buffer, format="JPEG")
        buffer.seek(0)

        # Step 6: Encode the final image back to base64
        final_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        return final_base64

    except Exception as e:
        raise RuntimeError(f"Error processing base64 image: {e}")
    

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
                url, timeout=client_timeout, json={"prompt": prompt, "uuid": uid}
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

file_names = ["preview.jpeg", "output.glb"]

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
            # normalize preview images
            processed_base64 = process_base64_image(synapse.out_prev)
            save_file(os.path.join(save_dir, 'preview.jpeg'), base64.b64decode(processed_base64))
            save_file(os.path.join(save_dir, 'output.glb'), decode_base64(synapse.out_glb, "glb"))
        except Exception as e:
            bt.logging.debug(f"Error saving synapse files: {e}")
