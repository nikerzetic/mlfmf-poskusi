import torch
from code2seq.model import Code2Seq


class Code2SeqResultsReader:

    def __init__(self, model: Code2Seq):
        self._id2label = {v: k for k, v in model.vocabulary.label_to_id.items()}

    def tensor2label(self, batch: torch.Tensor) -> list[str]:
        decoded_labels = []
        for id in batch:
            subtoken = self._id2label[id.item()]
            if subtoken == "<SOS>":
                continue
            if subtoken == "<EOS>":
                break
            decoded_labels.append(subtoken)
        return "|".join(decoded_labels)