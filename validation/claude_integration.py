import os
import json
import requests
import base64
import time
from pathlib import Path
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("CLAUDE_API_KEY", None)

BASE_DIR = './validation/output_images'

def get_image_description(image_path, max_retries=20):
    
    try:
        
        if api_key is None:
            raise ValueError("CLAUDE_API_KEY environment variable not set")

        start_time = time.time()
        
        # Read and encode image
        with open(image_path, 'rb') as image_file:
            image_data = base64.b64encode(image_file.read()).decode('utf-8')

        # API configuration
        url = "https://api.anthropic.com/v1/messages"
        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

        # Request body
        body = {
            "model": "claude-3-haiku-20240307",
            "max_tokens": 1024,
            "temperature": 0.1,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": image_data,
                            },
                        },
                        {
                            "type": "text",
                            "text": "If this image contains ANY text, respond with ONLY the exact phrase 'Texted Image' and nothing else. If there is NO text in the image, describe the physical object shown in one concise sentence suitable for 3D modeling. Focus on the object itself, not the photographic qualities."
                        }
                    ],
                }
            ],
        }

        retry_count = 0
        while retry_count < max_retries:
            try:
                response = requests.post(url, headers=headers, json=body)
                
                if response.status_code == 200:
                    response_data = response.json()
                    processing_time = time.time() - start_time
                    print(f"Request completed in {processing_time:.2f} seconds")
                    return response_data['content'][0]['text'].strip()
                
                # Handle rate limiting and other non-200 responses
                wait_time = 10 if response.status_code == 429 else 5
                print(f"Attempt {retry_count + 1}/{max_retries}: Received status code {response.status_code}. Waiting {wait_time} seconds...")
                time.sleep(wait_time)
                retry_count += 1
                
            except Exception as e:
                print(f"Attempt {retry_count + 1}/{max_retries}: Error: {str(e)}")
                time.sleep(5)
                retry_count += 1

        print(f"Failed to process {image_path} after {max_retries} retries")
        return None
    
    except Exception as e:
        print(f"Failed during Claude integration. error: {e}")


def get_render_img_descs():
    image_descs = []
    
    # Generate angles from 0 to 336 with step of 48
    angles = range(24, 360, 48)  # This will give [24, 72, ...]
    
    for angle in angles:
        image_path = f"{BASE_DIR}/image_{angle}.jpeg"
        desc = get_image_description(image_path)
        
        if desc:
            image_descs.append(desc)
        else:
            print(f"Failed to get description for angle {angle}")
    
    return image_descs


def get_prev_img_desc(prev_image_path):
    
    desc = get_image_description(prev_image_path)
    
    return desc
