from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef
from typing import Dict, List
from collections import deque
import bittensor as bt
import numpy as np


class MinerPerformanceTracker:
    """
    Tracks all recent miner performance to facilitate reward computation.
    """
    def __init__(self, store_last_n_predictions: int = 100):
        self.prediction_history: Dict[int, List[int]] = {}
        self.label_history: Dict[int, List[int]] = {}
        self.miner_addresses: Dict[int, str] = {}
        self.store_last_n_predictions = store_last_n_predictions

    def update(self, uid: int, prediction: int, label: int, miner_hotkey: str):
        """
        Update the miner prediction history
        """
        # Reset histories if miner is new or miner address has changed
        if uid not in self.prediction_history or self.miner_addresses[uid] != miner_hotkey:
            self.prediction_history[uid] = deque(maxlen=self.store_last_n_predictions)
            self.label_history[uid] = deque(maxlen=self.store_last_n_predictions)
            self.miner_addresses[uid] = miner_hotkey

        # Update histories
        self.prediction_history[uid].append(prediction)
        self.label_history[uid].append(label)

    def get_metrics(self, uid: int, n_predictions: int = 100):
        """
        Get the performance metrics for a miner based on their last n predictions
        """
        if n_predictions > self.store_last_n_predictions:
            bt.logging.warning(
                f"n_predictions {n_predictions} > store_last_n_predictions {self.store_last_n_predictions}.")

        recent_preds = self.prediction_history[uid][-n_predictions:]
        recent_labels = self.label_history[uid][-n_predictions:]        
        keep_idx = [i for i, p in enumerate(recent_preds) if p != -1]
        predictions = np.array([recent_preds[i] for i in keep_idx])
        labels = np.array([recent_labels[i] for i in keep_idx])

        accuracy = 0
        precision = 0
        recall = 0
        f1 = 0
        mcc = 0

        if len(labels) > 0 and len(predictions) > 0:
            # Calculate performance metrics
            try:
                accuracy = accuracy_score(labels, predictions)
                precision = precision_score(labels, predictions, zero_division=0)
                recall = recall_score(labels, predictions, zero_division=0)
                f1 = f1_score(labels, predictions, zero_division=0)
                # MCC requires at least two classes in labels
                mcc = matthews_corrcoef(labels, predictions) if len(np.unique(labels)) > 1 else 0.0
            except Exception as e: # TODO check for specific excpetion
                print('error in reward metric computation')
                print(e)
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'mcc': mcc
        }
