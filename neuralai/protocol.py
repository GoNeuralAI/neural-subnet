import typing
import bittensor as bt
import typing
import pydantic
from typing import List

# This is the protocol for the dummy miner and validator.
# It is a simple request-response protocol where the validator sends a request
# to the miner, and the miner responds with a dummy response.

class NATextSynapse(bt.Synapse):
    # Required request input, filled by sending dendrite caller.
    prompt_text: str = pydantic.Field(
        default="",
        title="Prompt",
        description="Text prompt for 3d generation"
    )
    negative_prompt: str = pydantic.Field(
        default="",
        title="Negative Prompt",
        description="Text negative prompt for 3d generation"
    )
    seed: int = pydantic.Field(
        default=0,
        title="Seed",
        description="Seed for 3d generation"
    )
    face_limit: int = pydantic.Field(
        default=90000,
        title="Face Limit",
        description="Face limit for 3d generation"
    )
    # Optional request output, filled by receiving axon.
    out_prev: str = pydantic.Field(
        default="prev",
        title="Preview Image",
        description="Base64 encoded preview image"
    )
    # output obj, supported format is .obj
    out_glb: str = pydantic.Field(
        default="glb",
        title="3d Object",
        description="3d object file"
    )
    timeout: int = pydantic.Field(
        default=80,
        title="Generation Timeout",
        description="3d generation timeout for synapse"
    )
    # S3 store address
    s3_addr: List[str] = []
    
    computed_body_hash: str = ""

    def deserialize(self) -> str:
        """
        Deserialize the miner response.

        Returns:
        - List[Image.Image]: The deserialized response, which is a list of several images measured in different axis
        """
        return self.out_obj
    
class NAImageSynapse(bt.Synapse):
    # Required request input, filled by sending dendrite caller.
    # For our app and end users' requests
    prompt_image: str = ""
    # Optional request output, filled by receiving axon.
    out_obj: str = "obj"
    
    timeout: int = 100
    
    computed_body_hash: str = pydantic.Field("", title="Computed Body Hash", frozen=False)

    def deserialize(self) -> str:
        """
        Deserialize the miner response.

        Returns:
        - List[Image.Image]: The deserialized response, which is a list of several images measured in different axis
        """
        return self.out_obj

class NAStatus(bt.Synapse):
    status: str = pydantic.Field(
        default="",
        title="Miner Status",
        description="Current status of miner"
    )
    sn_version: int = pydantic.Field(
        default=0,
        title="Subnet Version",
        description="Subnet version of the neuron sending synapse"
    )
    computed_body_hash: str = ""
    
    def deserialize(self) -> str:
        """
        Deserialize the miner response.

        Returns:
        - List[Image.Image]: The deserialized response, which is a list of several images measured in different axis
        """
        return self.status