import typing
import bittensor as bt
import typing
import pydantic

# This is the protocol for the dummy miner and validator.
# It is a simple request-response protocol where the validator sends a request
# to the miner, and the miner responds with a dummy response.

class NATextSynapse(bt.Synapse):
    # Required request input, filled by sending dendrite caller.
    prompt_text: str = ""
    # Optional request output, filled by receiving axon.
    out_obj: str = "obj"
    # Query response timeout
    timeout: int = 100
    
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
    status: str = ""
    computed_body_hash: str = ""
    
    def deserialize(self) -> str:
        """
        Deserialize the miner response.

        Returns:
        - List[Image.Image]: The deserialized response, which is a list of several images measured in different axis
        """
        return self.status