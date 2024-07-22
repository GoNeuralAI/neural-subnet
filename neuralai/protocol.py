# The MIT License (MIT)
# Copyright © 2024 NeuralAI

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

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