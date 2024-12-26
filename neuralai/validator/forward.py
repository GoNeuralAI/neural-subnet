import time
import asyncio
import datetime
import numpy as np
import bittensor as bt

from neuralai.protocol import NATextSynapse
from neuralai.validator.reward import (
    get_rewards, normalize
)
from neuralai.utils.uids import get_synthetic_forward_uids, get_organic_forward_uids
from neuralai.validator import utils
from neuralai import __version__ as version
import traceback


async def handle_response(response, uid, config, nas_prompt_text, timeout):
    try:
        utils.save_synapse_files(response, uid)
    except ValueError as e:
        print(f"Error saving files for response {uid}: {e}")

    result = await utils.validate(
        config.validation.endpoint, nas_prompt_text, int(uid), timeout=timeout
    )
    process_time = response.dendrite.process_time
    return result['score'], (process_time if process_time and process_time > 10 else 0)


async def forward_synthetic(self, synapse: NATextSynapse = None) -> NATextSynapse:
    """
    The forward function is called by the validator every time step.

    It is responsible for querying the network and scoring the responses.

    - Forwarding requests to miners in multiple threads to ensure total time is around 300 seconds. In each thread, we do:
        - Calculating rewards if needed
        - Updating scores based on rewards
        - Saving the state
    - Normalize weights based on incentive_distribution
    - SET WEIGHTS!
    - Sleep for 300 seconds if needed
    """
    if self.status == "validation":
        while self.status != "validation":
            time.sleep(10)
    self.status = "validation"
    start_time = time.time()
    loop_time = self.config.neuron.task_period
    timeout = loop_time / 5
    val_scores = []

    if not self.config.wandb.off:
        today = datetime.date.today()
        if self.wandb_manager.wandb_start != today:
            self.wandb_manager.wandb.finish()
            self.wandb_manager.init_wandb()

    try:
        bt.logging.info("========================== Sending the synthetic synapse ============================")
        bt.logging.info(f"New Epoch v{version}")
        bt.logging.info("Checking Available Miners.....")

        avail_uids = get_synthetic_forward_uids(self, self.config.neuron.synthetic_challenge_count)
        bt.logging.info(f"Listed Miners Are: {avail_uids}")

        forward_uids = self.miner_manager.get_miner_status(uids=avail_uids)

        if len(forward_uids) == 0:
            bt.logging.warning("No Miners Are Available for synthetic synsapse!")
            val_scores = [0 for _ in avail_uids]
            self.update_scores(val_scores, avail_uids)
        else:
            bt.logging.info(f"Forward uids are: {forward_uids}")

            task = await self.task_manager.prepare_task()
            nas = NATextSynapse(prompt_text=task, timeout=timeout)

            if nas.prompt_text:
                process_time = []
                time_rate = self.config.validator.time_rate

                bt.logging.info(f"======== Current Task Prompt: {task} ========")

                responses = self.dendrite.query(
                    axons=[self.metagraph.axons[uid] for uid in forward_uids],
                    synapse=nas,
                    timeout=timeout,
                    deserialize=False,
                )
                bt.logging.info("Responses Received")

                start_vali_time = time.time()

                tasks = [
                    handle_response(
                        response,
                        forward_uids[index],
                        self.config,
                        nas.prompt_text,
                        loop_time - timeout
                    )
                    for index, response in enumerate(responses)
                ]
                results = await asyncio.gather(*tasks)

                val_scores, process_time = zip(*results)

                # Update rewards and scores
                scores = get_rewards(val_scores, avail_uids, forward_uids)
                f_val_scores = normalize(scores)

                bt.logging.info('-' * 40)
                bt.logging.info("=== 3D Object Validation Scores ===", np.round(scores, 3))

                scores = get_rewards(process_time, avail_uids, forward_uids)
                f_time_scores = normalize(scores)

                # Considered with the generation time score 0.1
                final_scores = [
                    s * (1 - time_rate) + time_rate * (1 - t) if t else s
                    for t, s in zip(f_time_scores, f_val_scores)
                ]

                bt.logging.info("=== Total Scores ===", np.round(final_scores, 3))
                bt.logging.info('-' * 40)

                self.update_scores(final_scores, avail_uids)

                bt.logging.info(f"Scoring Taken Time: {time.time() - start_vali_time:.1f}s")
            else:
                bt.logging.error("No prompt is ready yet")

            taken_time = time.time() - start_time

            if taken_time < loop_time:
                bt.logging.info(f"== Taken Time for synthetic synapse: {taken_time:.1f}s | Sleeping For {loop_time - taken_time:.1f}s ==")
                bt.logging.info('=' * 60)
                time.sleep(loop_time - taken_time)
        self.status = "idel"

    except Exception as e:
        self.status = "idle"
        bt.logging.error(traceback.format_exc())


async def forward_organic(self, synapse: NATextSynapse = None) -> NATextSynapse:
    try:
        self.status = "validation"
        start_time = time.time()
        loop_time = self.config.neuron.task_period
        timeout = loop_time / 5
        if synapse.prompt_text is None:
            raise Exception("None prompt of organic synapse.")
        print("timeout = ", timeout)
        nas = NATextSynapse(prompt_text=synapse.prompt_text, timeout=timeout)

        bt.logging.info("============================ Sending the organic synapse ============================")
        bt.logging.info("Finding the available top miners...")
        avail_uids = get_organic_forward_uids(self, self.config.neuron.organic_challenge_count)
        bt.logging.info(f"Listed Miners Are: {avail_uids}")

        ping_uids = self.miner_manager.get_miner_status(uids=avail_uids)
        # print("Passed here")
        # sorted_uids = sorted(ping_uids, key=lambda uid: avail_uids.index(uid))
        sorted_uids = ping_uids
        if len(sorted_uids) < 1:
            bt.logging.info("There is no available miners for organic synapse")
            self.status = "idle"
        else:
            query_count = min(self.config.neuron.organic_query_count, len(sorted_uids))
            forward_uids = sorted_uids[:query_count]

            bt.logging.info(f"Current organic synapse prompt: {synapse.prompt_text}")
            bt.logging.info(f"Forward uids are: {forward_uids}")

            responses = self.dendrite.query(
                axons=[self.metagraph.axons[uid] for uid in forward_uids],
                synapse=nas,
                timeout=timeout,
                deserialize=False,
            )

            bt.logging.info("Responses Received")
            taken_time = time.time() - start_time
            bt.logging.info(f"== Taken Time for synthetic synapse: {taken_time:.1f}s")
            bt.logging.info('=' * 60)
            self.status = "idle"
            for response in responses:
                if response is not None:
                            return response
        
    except Exception as e:
        self.status = "idle"
        bt.logging.error(traceback.format_exc())