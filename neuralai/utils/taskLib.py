import httpx
import os
from dotenv import load_dotenv
import bittensor as bt

# Load environment variables from .env file
# load_dotenv()
NEURAL_API_URL = "https://api.goneural.ai/v1/validator-query"

class taskLib:
    async def get_task(self):
        if NEURAL_API_URL is None:
            bt.logging.error("NEURAL_API_URL environment variable is not set.")
            return None
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(NEURAL_API_URL)
                response.raise_for_status()  # Raise an error for bad responses
                prompts = response.json()  # Assuming the response is in JSON format
                return prompts
        except httpx.RequestError as e:
            bt.logging.error(f"Error fetching prompts: {e}")
            return None
