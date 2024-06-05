from typing import Optional, Union

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
        _, batch_size = target.shape
        # [batch size; vocab size; seq length]
        _logits = logits.permute(1, 2, 0)
        # [batch size; seq length]
        _labels = target.permute(1, 0)
        # [batch size; seq length]
        loss = nn.functional.cross_entropy(_logits, _labels, reduction="none")
        if self.__pad_idx is not None:
            # [batch size; seq length]
            mask = _labels != self.__pad_idx
            seq_len: Union[int, Tensor] = mask.sum(-1)
            # [batch size; seq length]
            example_loss = loss * mask
        else:
            # [batch size; seq length]
            example_loss = loss
            seq_len = example_loss.shape[1] 

        if self.__reduction is None:
            return example_loss
        elif self.__reduction == "seq-mean":
            return (example_loss.sum(-1) / seq_len).mean()
        elif self.__reduction == "seq-sum":
            return (example_loss.sum(-1) / seq_len).sum()
        elif self.__reduction == "batch-mean":
            return example_loss[~example_loss.isnan()].sum() / batch_size
        else:
            raise NotImplementedError(f"Unknown reduction: {self.__reduction}")