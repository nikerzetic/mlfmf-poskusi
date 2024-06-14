from typing import Optional, Union

import torch
from torch import nn, Tensor


class SequenceCrossEntropyLoss(nn.CrossEntropyLoss):
    """Calculate cross entropy loss with ignoring PAD index.

    :param pad_idx: an optional index of padding in target sequence.
    :param reduction: rule to reduce computed loss over batch.
    """

    def __init__(self, pad_idx: Optional[int] = None, reduction: Optional[str] = "mean"):
        super().__init__()
        self.__pad_idx = pad_idx
        self.__reduction = reduction

    def forward(self, logits: Tensor, target: Tensor) -> Tensor:
        """Calculated loss for given logits and target.

        :param logits: tensor with logits with shape [seq len; batch size; vocab size]
        :param target: tensor with target classes with shape [seq len; batch size]
        :return:
        """
        # _, batch_size = target.shape
        # [batch size; vocab size; seq length]
        _logits = logits.permute(1, 2, 0)
        # [batch size; seq length]
        _labels = target.permute(1, 0)
        # [batch size; seq length]
        _loss = nn.functional.cross_entropy(_logits, _labels, reduction="none")
        if self.__pad_idx is not None:
            # [batch size; seq length]
            mask = _labels != self.__pad_idx
            seq_len: Union[int, Tensor] = mask.sum(-1)
            # [batch size; seq length]
            example_loss = _loss * mask
            # [batch size; seq length]

            #HACK: We punish the model for incorerctly predicting pad
            # predicted_pad = _logits.argmax(1) == self.__pad_idx
            # example_loss += predicted_pad*(~mask)
            
            #HACK: We only consider non-empty rows
            empty_row = torch.all(_labels == self.__pad_idx, dim=1)
            example_loss = example_loss[~empty_row]
        else:
            # [batch size; seq length]
            example_loss = _loss
            seq_len = example_loss.shape[1] 

        batch_size, _ = example_loss.shape
        example_loss[example_loss.isnan()] = 10

        if self.__reduction is None:
            loss = example_loss
        elif self.__reduction == "seq-mean":
            loss = (example_loss.sum(-1) / seq_len).mean()
        elif self.__reduction == "seq-sum":
            loss = (example_loss.sum(-1) / seq_len).sum()
        elif self.__reduction == "batch-mean":
            loss = example_loss.sum() 
            if not batch_size == 0:
                loss /= batch_size
        else:
            raise NotImplementedError(f"Unknown reduction: {self.__reduction}")
        
        if loss.isnan().item(): #TODO: fix so that it doesn't break the training loop
            raise ValueError("Got a loss nan")
        
        if target is None: #BUG: we never get here, though we should in the val step already
            raise ValueError("Testing if this is the problem")

        return loss