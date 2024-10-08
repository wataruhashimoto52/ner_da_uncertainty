import logging
from typing import Union

import numpy as np
import torch.nn as nn
from transformers import PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerFast

logger = logging.getLogger(__name__)


def apply_dropout(m: nn.Module):
    if type(m) == nn.Dropout:
        m.train()


class EarlyStoppingForTransformers:
    def __init__(self, path: str, patience: int = 5, verbose: bool = False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.path = path

    def __call__(
        self,
        val_loss: float,
        model: PreTrainedModel,
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
        save_flag: bool = True,
    ) -> bool:
        score = val_loss

        if self.best_score is None:
            self.best_score = score
            self.checkpoint(
                val_loss, model, tokenizer, save_flag
            )
            return True

        elif score < self.best_score:
            self.counter += 1
            if self.verbose:
                logger.info(
                    f"EarlyStopping counter: {self.counter} out of {self.patience}"
                )
            if self.counter >= self.patience:
                self.early_stop = True

            return False

        else:
            self.best_score = score
            self.counter = 0
            return True

    def checkpoint(
        self,
        val_loss: float,
        model: PreTrainedModel,
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
        save_flag: bool = True,
    ):
        if self.verbose:
            logger.info(
                f"Validation score updated. ({self.val_loss_min:.6f} --> {val_loss:.6f})."
            )
        # Save Model
        if save_flag:
            logger.info("Saving model...")
            model.save_pretrained(self.path)
            tokenizer.save_pretrained(self.path)

        self.val_loss_min = val_loss
