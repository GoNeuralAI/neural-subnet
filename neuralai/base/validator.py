# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# TODO(developer): Set your name
# Copyright © 2023 <your name>

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


import copy
import numpy as np
import asyncio
import argparse
import threading
import bittensor as bt
import math
from collections import Counter

from typing import List, Union
from traceback import print_exception

from neuralai.base.neuron import BaseNeuron
from neuralai.base.utils.weight_utils import process_weights_for_netuid, convert_weights_and_uids_for_emit  # TODO: Replace when bittensor switches to numpy
from neuralai.mock import MockDendrite
from neuralai.utils.config import add_validator_args


class BaseValidatorNeuron(BaseNeuron):
    """
    Base class for Bittensor validators. Your validator should inherit from this class.
    """

    neuron_type: str = "ValidatorNeuron"

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser):
        super().add_args(parser)
        add_validator_args(cls, parser)

    def __init__(self, config=None):
        super().__init__(config=config)

        # Save a copy of the hotkeys to local memory.
        self.hotkeys = copy.deepcopy(self.metagraph.hotkeys)

        self.status = "idle"
        
        # Dendrite lets us send messages to other nodes (axons) in the network.
        if self.config.mock:
            self.dendrite = MockDendrite(wallet=self.wallet)
        else:
            self.dendrite = bt.dendrite(wallet=self.wallet)
        bt.logging.info(f"Dendrite: {self.dendrite}")

        # Set up initial scoring weights for validation
        bt.logging.info("Building validation weights.")
        self.base_scores = np.zeros(
            self.metagraph.n, dtype=np.float32
        )
        self.scores = np.zeros(
            self.metagraph.n, dtype=np.float32
        )

        # Init sync with the network. Updates the metagraph.
        self.sync()

        # Serve axon to enable external connections.
        if not self.config.neuron.axon_off:
            self.serve_axon()
        else:
            bt.logging.warning("axon off, not serving ip to chain.")

        # Create asyncio event loop to manage async tasks.
        self.loop = asyncio.get_event_loop()

        # Instantiate runners
        self.should_exit: bool = False
        self.is_running: bool = False
        self.thread: Union[threading.Thread, None] = None
        self.lock = asyncio.Lock()

    def serve_axon(self):
        """Serve axon to enable external connections."""

        bt.logging.info("serving ip to chain...")
        try:
            self.axon = bt.axon(wallet=self.wallet, config=self.config)

            self.axon.attach(
                forward_fn=self.forward_fn,
                blacklist_fn=self.whitelist_fn_query,
                priority_fn=self.priority_fn_query,
            ).attach(
                forward_fn=self.forward_status,
                blacklist_fn=self.whitelist_fn_status,
                priority_fn=self.priority_fn_status,
            )
            
            try:
                self.axon.serve(netuid=self.config.netuid, subtensor=self.subtensor)
                self.axon.start()
                bt.logging.info(
                    f"Running validator {self.axon} on network: {self.config.subtensor.chain_endpoint} with netuid: {self.config.netuid}"
                )
            except Exception as e:
                bt.logging.error(f"Failed to serve Axon with exception: {e}")
                pass

        except Exception as e:
            bt.logging.error(
                f"Failed to create Axon initialize with exception: {e}"
            )
            pass

    async def concurrent_forward(self):
        coroutines = [
            self.forward_synthetic()
            for _ in range(self.config.neuron.num_concurrent_forwards)
        ]
        await asyncio.gather(*coroutines, return_exceptions=True)

    def run(self):
        """
        Initiates and manages the main loop for the miner on the Bittensor network. The main loop handles graceful shutdown on keyboard interrupts and logs unforeseen errors.

        This function performs the following primary tasks:
        1. Check for registration on the Bittensor network.
        2. Continuously forwards queries to the miners on the network, rewarding their responses and updating the scores accordingly.
        3. Periodically resynchronizes with the chain; updating the metagraph with the latest network state and setting weights.

        The essence of the validator's operations is in the forward function, which is called every step. The forward function is responsible for querying the network and scoring the responses.

        Note:
            - The function leverages the global configurations set during the initialization of the miner.
            - The miner's axon serves as its interface to the Bittensor network, handling incoming and outgoing requests.

        Raises:
            KeyboardInterrupt: If the miner is stopped by a manual interruption.
            Exception: For unforeseen errors during the miner's operation, which are logged for diagnosis.
        """

        # Check that validator is registered on the network.
        self.sync()

        bt.logging.info(f"Validator starting at block: {self.block}")

        # This loop maintains the validator's operations until intentionally stopped.
        while True:
            try:
                # bt.logging.info(f"step({self.step}) block({self.block}) last_update({self.metagraph.last_update[self.uid]})")

                # Run forward.
                try:
                    self.loop.run_until_complete(self.concurrent_forward())
                except Exception as err:
                    bt.logging.error(f"Error during validation: {str(err)}")
                    bt.logging.debug(str(print_exception(type(err), err, err.__traceback__)))

                # Check if we should exit.
                if self.should_exit:
                    break

                # Sync metagraph and potentially set weights.
                self.sync()

                self.step += 1

            # If someone intentionally stops the validator, it'll safely terminate operations.
            except KeyboardInterrupt:
                self.axon.stop()
                bt.logging.success("Validator killed by keyboard interrupt.")
                exit()

            # In case of unforeseen errors, the validator will log the error and continue operations.
            except Exception as err:
                bt.logging.error(f"Error during validation: {str(err)}")
                bt.logging.debug(
                    str(print_exception(type(err), err, err.__traceback__))
                )

    def run_in_background_thread(self):
        """
        Starts the validator's operations in a background thread upon entering the context.
        This method facilitates the use of the validator in a 'with' statement.
        """
        if not self.is_running:
            bt.logging.debug("Starting validator in background thread.")
            self.should_exit = False
            self.thread = threading.Thread(target=self.run, daemon=True)
            self.thread.start()
            self.is_running = True
            bt.logging.debug("Started")

    def stop_run_thread(self):
        """
        Stops the validator's operations that are running in the background thread.
        """
        if self.is_running:
            bt.logging.debug("Stopping validator in background thread.")
            self.should_exit = True
            self.thread.join(5)
            self.is_running = False
            bt.logging.debug("Stopped")

    def __enter__(self):
        self.run_in_background_thread()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """
        Stops the validator's background operations upon exiting the context.
        This method facilitates the use of the validator in a 'with' statement.

        Args:
            exc_type: The type of the exception that caused the context to be exited.
                      None if the context was exited without an exception.
            exc_value: The instance of the exception that caused the context to be exited.
                       None if the context was exited without an exception.
            traceback: A traceback object encoding the stack trace.
                       None if the context was exited without an exception.
        """
        if self.is_running:
            bt.logging.debug("Stopping validator in background thread.")
            self.should_exit = True
            self.thread.join(5)
            self.is_running = False
            bt.logging.debug("Stopped")

    def set_weights(self):
        """
        Sets the validator weights to the metagraph hotkeys based on the scores it has received from the miners. The weights determine the trust and incentive level the validator assigns to miner nodes on the network.
        """
        # Create a list of (id, score) pairs
        bt.logging.info(f"base_scores: {self.base_scores}")
        id_score_pairs = list(enumerate(self.base_scores))
        
        sorted_pairs = sorted(id_score_pairs, key=lambda x: x[1], reverse=True)
        
        # Calculate ranks (handling ties)
        ranks = []
        current_rank = 1
        previous_score = None
        for i, (id, score) in enumerate(sorted_pairs):
            if score != previous_score:
                current_rank = i
            ranks.append((id, current_rank, score))
            previous_score = score
        
        # Sort back to original order
        ranks.sort(key=lambda x: x[0])
        
        # self.scores = [(math.exp(-0.03 * rank) if score > 0 else 0) for id, rank, score in ranks]
        self.scores = [(score ** 8 if score > 4e-1 else 0) for score in self.base_scores]
        
        bt.logging.info(f"scores: {self.scores}")

        # Check if self.scores contains any NaN values and log a warning if it does.
        if np.isnan(self.scores).any():
            bt.logging.warning(
                f"Scores contain NaN values. This may be due to a lack of responses from miners, or a bug in your reward functions."
            )

        # Calculate the average reward for each uid across non-zero values.
        # Replace any NaN values with 0.
        # Compute the norm of the scores
        norm = np.linalg.norm(self.scores, ord=1, axis=0, keepdims=True)

        # Check if the norm is zero or contains NaN values
        if np.any(norm == 0) or np.isnan(norm).any():
            norm = np.ones_like(norm)  # Avoid division by zero or NaN

        # Compute raw_weights safely
        raw_weights = self.scores / norm

        bt.logging.debug("raw_weights", raw_weights)
        bt.logging.debug("raw_weight_uids", str(self.metagraph.uids.tolist()))
        # Process the raw weights to final_weights via subtensor limitations.
        (
            processed_weight_uids,
            processed_weights,
        ) = process_weights_for_netuid(
            uids=self.metagraph.uids,
            weights=raw_weights,
            netuid=self.config.netuid,
            subtensor=self.subtensor,
            metagraph=self.metagraph,
        )
        bt.logging.debug("processed_weights", processed_weights)
        bt.logging.debug("processed_weight_uids", processed_weight_uids)

        # Convert to uint16 weights and uids.
        (
            uint_uids,
            uint_weights,
        ) = convert_weights_and_uids_for_emit(
            uids=processed_weight_uids, weights=processed_weights
        )
        bt.logging.debug("uint_weights", uint_weights)
        bt.logging.debug("uint_uids", uint_uids)

        # Set the weights on chain via our subtensor connection.
        result, msg = self.subtensor.set_weights(
            wallet=self.wallet,
            netuid=self.config.netuid,
            uids=uint_uids,
            weights=uint_weights,
            wait_for_finalization=False,
            wait_for_inclusion=False,
            version_key=self.spec_version,
        )
        if result is True:
            bt.logging.info("set_weights on chain successfully!")
        else:
            bt.logging.error("set_weights failed", msg)

    def resync_metagraph(self):
        """Resyncs the metagraph and updates the hotkeys and moving averages based on the new metagraph."""
        bt.logging.info("resync_metagraph()")

        # Copies state of metagraph before syncing.
        previous_metagraph = copy.deepcopy(self.metagraph)

        # Sync the metagraph.
        self.metagraph.sync(subtensor=self.subtensor)

        # Check if the metagraph axon info has changed.
        if previous_metagraph.axons == self.metagraph.axons:
            return

        bt.logging.info(
            "Metagraph updated, re-syncing hotkeys, dendrite pool and moving averages"
        )
        # Zero out all hotkeys that have been replaced.
        for uid, hotkey in enumerate(self.hotkeys):
            if hotkey != self.metagraph.hotkeys[uid]:
                self.scores[uid] = 0  # hotkey has been replaced

        # Check to see if the metagraph has changed size.
        # If so, we need to add new hotkeys and moving averages.
        if len(self.hotkeys) < len(self.metagraph.hotkeys):
            # Update the size of the moving average scores.
            new_moving_average = np.zeros((self.metagraph.n))
            min_len = min(len(self.hotkeys), len(self.scores))
            new_moving_average[:min_len] = self.scores[:min_len]
            self.scores = new_moving_average

        # Update the hotkeys.
        self.hotkeys = copy.deepcopy(self.metagraph.hotkeys)

    def check_serving_axon(self, metagraph: "bt.metagraph.Metagraph", uid: int, vpermit_tao_limit: int) -> bool:
        # Filter non serving axons.
        if metagraph.validator_permit[uid]:
            if metagraph.S[uid] >= vpermit_tao_limit:
                return True
            
        if not metagraph.axons[uid].is_serving:
            return False
        # Available otherwise.
        return True


    def update_scores(self, rewards: np.ndarray, uids: List[int]):
        """Performs exponential moving average on the scores based on the rewards received from the miners."""

        # Check if rewards contains NaN values.
        if np.isnan(rewards).any():
            bt.logging.warning(f"NaN values detected in rewards: {rewards}")
            # Replace any NaN values in rewards with 0.
            rewards = np.nan_to_num(rewards, nan=0)

        # Ensure rewards is a numpy array.
        rewards = np.asarray(rewards)

        # Check if `uids` is already a numpy array and copy it to avoid the warning.
        if isinstance(uids, np.ndarray):
            uids_array = uids.copy()
        else:
            uids_array = np.array(uids)

        # Handle edge case: If either rewards or uids_array is empty.
        if rewards.size == 0 or uids_array.size == 0:
            bt.logging.info(f"rewards: {rewards}, uids_array: {uids_array}")
            bt.logging.warning("Either rewards or uids_array is empty. No updates will be performed.")
            return

        # Check if sizes of rewards and uids_array match.
        if rewards.size != uids_array.size:
            raise ValueError(f"Shape mismatch: rewards array of shape {rewards.shape} "
                             f"cannot be broadcast to uids array of shape {uids_array.shape}")

        # Compute forward pass rewards, assumes uids are mutually exclusive.
        # shape: [ metagraph.n ]
        uids_list = self.metagraph.uids.tolist()

        # Calculate how many elements to add
        count_to_add = len(uids_list) - len(self.base_scores)

        # Extend base_scores if needed
        if count_to_add > 0:
            additional_zeros = np.zeros(count_to_add, dtype=np.float32)
            # Concatenate the two arrays
            self.base_scores = np.concatenate((self.base_scores, additional_zeros))

        # Create list of non-serving UIDs
        non_serving_uids = []
        for uid in range(self.metagraph.n.item()):
            if not self.check_serving_axon(self.metagraph, uid, self.config.neuron.vpermit_tao_limit):
                non_serving_uids.append(uid)
        
        bt.logging.debug("Giving penalty scores to non-serving uids...")
        bt.logging.debug(f"Non-serving uids: {non_serving_uids}")

        # Initialize scattered rewards
        scattered_rewards: np.ndarray = np.full_like(self.base_scores, -1)
        scattered_rewards[uids_array] = rewards
        
        # Apply zero scores to non-serving miners
        scattered_rewards[non_serving_uids] = 0
        bt.logging.debug(f"Final rewards (including penalties): {scattered_rewards}")

        # Update base_scores with rewards produced by this step.
        # shape: [ metagraph.n ]
        alpha: float = self.config.neuron.moving_average_alpha
        for i in range(len(self.base_scores)):
            if scattered_rewards[i] != -1:
                self.base_scores[i] = alpha * (0 if scattered_rewards[i] < 0.1 else scattered_rewards[i]) + (1 - alpha) * self.base_scores[i]

        self.base_scores = np.where(self.base_scores < 4e-2, 0, self.base_scores)
        
        bt.logging.info(f"Updated moving avg base_scores: {self.base_scores}")

    def save_state(self):
        """Saves the state of the validator to a file."""
        bt.logging.info("Saving validator state.")

        # Save the state of the validator to file.
        # self.base_scores = np.where(self.base_scores < 1e-3, 0, self.base_scores)
        np.savez(
            self.config.neuron.full_path + "/state.npz",
            step=self.step,
            scores=self.base_scores,
            hotkeys=self.hotkeys,
        )

    def load_state(self):
        """Loads the state of the validator from a file."""

        # Load the state of the validator from file.
        try:
            state = np.load(self.config.neuron.full_path + "/state.npz")
            bt.logging.info(f"Loading validator state.{state['scores']}")
            self.step = state["step"]
            for i in range(len(state["scores"])):
                self.base_scores[i] = float(state["scores"][i])
            self.hotkeys = state["hotkeys"]
        except Exception as e:
            print("Couldn't find save file!")
