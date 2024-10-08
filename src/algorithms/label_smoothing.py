from typing import Final

import torch
import torch.nn as nn


class LabelSmoother:
    ignore_index: Final[int] = -100

    def __init__(self, reduction: str, epsilon: float = 0.1) -> None:
        self.reduction = reduction
        self.epsilon = epsilon

    def __call__(
        self, logits: torch.Tensor, labels: torch.Tensor, shift_labels: bool = False
    ) -> torch.Tensor:
        if shift_labels:
            logits = logits[..., :-1, :].contiguous()
            labels = labels[..., 1:].contiguous()

        log_probs = -nn.functional.log_softmax(logits, dim=-1)
        if labels.dim() == log_probs.dim() - 1:
            labels = labels.unsqueeze(-1)

        padding_mask = labels.eq(self.ignore_index)
        # In case the ignore_index is -100, the gather will fail, so we replace labels by 0. The padding_mask
        # will ignore them in any case.
        labels = torch.clamp(labels, min=0)
        nll_loss = log_probs.gather(dim=-1, index=labels)
        # works for fp16 input tensor too, by internally upcasting it to fp32
        smoothed_loss = log_probs.sum(dim=-1, keepdim=True, dtype=torch.float32)

        nll_loss.masked_fill_(padding_mask, 0.0)
        smoothed_loss.masked_fill_(padding_mask, 0.0)

        # Take the mean over the label dimensions, then divide by the number of active elements (i.e. not-padded):
        num_active_elements = padding_mask.numel() - padding_mask.long().sum()

        if self.reduction == "mean":
            nll_loss = nll_loss.sum() / num_active_elements
            smoothed_loss = smoothed_loss.sum() / (
                num_active_elements * log_probs.shape[-1]
            )

        elif self.reduction == "sum":
            nll_loss = nll_loss.sum()
            smoothed_loss = smoothed_loss.sum()

        elif self.reduction == "none":
            pass

        else:
            raise NotImplementedError()

        return (1 - self.epsilon) * nll_loss + self.epsilon * smoothed_loss
