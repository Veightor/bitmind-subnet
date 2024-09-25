# The MIT License (MIT)
# Copyright © 2024
# developer: kenobijon
# Copyright © 2024 Ken Jon Miyachi

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

from typing import List, Dict
import bittensor as bt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef
)


def count_penalty(y_pred: float) -> float:
    # Penalize if prediction is not within [0, 1]
    bad = (y_pred < 0.0) or (y_pred > 1.0)
    return 0.0 if bad else 1.0

def get_rewards(
        label: float,
        responses: List[float],
        uids: List[int],
        axons: List[bt.axon],
        performance_tracker,
        num_prev_preds: int = 100
    ) -> np.array:
    """
    Returns an array of rewards for the given label and miner responses.

    Args:
    - label (float): The true label (1.0 for fake, 0.0 for real).
    - responses (List[float]): A list of responses from the miners.
    - uids (List[int]): List of miner UIDs.
    - axons (List[bt.axon]): List of miner axons.
    - performance_tracker (MinerPerformanceTracker): Tracks historical performance metrics per miner.
    - num_prev_preds (int): Number of previous predictions to consider for metrics calculation.

    Returns:
    - np.array: An array of rewards for the given label and responses.
    """
    miner_rewards = []
    for axon, uid in zip(axons, uids):
        try:
            miner_hotkey = axon.hotkey
            
            # Check if the uid is within the range of responses
            if uid >= len(responses):
                bt.logging.error(f"UID {uid} is out of range for responses. Skipping this miner.")
                miner_rewards.append(0.0)
                continue

            pred_prob = responses[uid]
            
            # Apply penalty if prediction is invalid
            penalty = count_penalty(pred_prob)
            pred = int(np.round(pred_prob))
            true_label = int(label)

            # Update miner's performance history
            performance_tracker.update(uid, pred, true_label, miner_hotkey)

            # Get historical performance metrics
            metrics = performance_tracker.get_metrics(uid)
            
            # Calculate ramp-up factor (reaches 1.0 at num_prev_preds predictions)
            num_predictions = len(performance_tracker.prediction_history[uid])
            ramp_up_factor = min(num_predictions / num_prev_preds, 1.0)

            # Calculate current reward as a linear combination of metrics
            reward = (
                0.2 * metrics['accuracy'] +
                0.2 * metrics['precision'] +
                0.2 * metrics['recall'] +
                0.2 * metrics['f1_score'] +
                0.2 * metrics['mcc']
            )

            # Apply penalty and ramp-up factor
            reward *= ramp_up_factor * penalty

            miner_rewards.append(reward)

            # Optionally, log metrics for debugging
            bt.logging.debug(f"""
            Miner {uid} Performance:
            Accuracy:  {metrics['accuracy']:.4f}
            Precision: {metrics['precision']:.4f}
            Recall:    {metrics['recall']:.4f}
            F1 Score:  {metrics['f1_score']:.4f}
            MCC:       {metrics['mcc']:.4f}
            Penalty:   {penalty:.4f}
            Reward:    {reward:.4f}
            """)
            
        except Exception as e:
            bt.logging.error(f"Couldn't calculate reward for miner {uid}, prediction: {responses[uid] if uid < len(responses) else 'N/A'}, label: {label}")
            bt.logging.exception(e)
            miner_rewards.append(0.0)

    return np.array(miner_rewards)


def old_get_rewards(
        label: float,
        responses: List,
    ) -> np.array:
    """
    Returns a tensor of rewards for the given query and responses.

    Args:
    - label (float): 1 if image was fake, 0 if real.
    - responses (List[float]): A list of responses from the miners.

    Returns:
    - np.array: A tensor of rewards for the given query and responses.
    """
    miner_rewards = []
    for uid in range(len(responses)):
        try:
            pred = responses[uid]
            reward = 1. if np.round(pred) == label else 0.
            reward *= count_penalty(pred)
            miner_rewards.append(reward)

        except Exception as e:
            bt.logging.error("Couldn't count miner reward for {}, his predictions = {} and his labels = {}".format(
                uid, responses[uid], label))
            bt.logging.exception(e)
            miner_rewards.append(0)

    return np.array(miner_rewards)
