import torch
import helpers
import entries_extractor as ee
from code2seq.model import Code2Seq


class Code2SeqResultsReader:

    def __init__(self, model: Code2Seq):
        # TODO: reader shouldn't be model-dependent, but the other way around
        self._label2id = model.vocabulary.label_to_id
        self._id2label = {v: k for k, v in model.vocabulary.label_to_id.items()}

    def tensor2label(self, batch: torch.Tensor) -> str:
        # TODO: figure out what this is actually doing - what it returns
        decoded_labels = []
        for id in batch:
            subtoken = self._id2label[id.item()]
            if subtoken == "<SOS>":
                continue
            if subtoken == "<EOS>":
                break
            decoded_labels.append(subtoken)
        return "|".join(decoded_labels)
    
    def raw2label(self, file: str):
        # TODO: fix implementation - don't read from file
        f = open(file, "r", encoding="utf-8")
        raw2label = {}
        f.readline()
        for line in f:
            name = line.split("\t")[0]
            # TODO: the following should be a standalone reader passed as parameter 
            label = helpers.replace_unicode_with_latex(ee.format_as_label(name))
            label_parts = label.split("|")
            new_label_parts = []
            for part in label_parts:
                if part in self._label2id:
                    new_label_parts.append(part)
                    continue
                new_label_parts.append("<UNK>")
            raw2label[name] = "|".join(new_label_parts)
        f.close()
        return raw2label
