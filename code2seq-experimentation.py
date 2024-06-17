import pytorch_lightning as L
import argparse
import sys
import torch
import tqdm

from typing import *
from omegaconf import OmegaConf, DictConfig
from code2seq.model import Code2Class, Code2Seq
from code2seq.data.path_context_data_module import PathContextDataModule
from my_code2seq.helpers import Code2SeqResultsReader


def model_setup(config: DictConfig, data_module: PathContextDataModule):

    model = Code2Seq(
        config.model,
        config.optimizer,
        data_module.vocabulary,
        config.train.teacher_forcing,
    )

    return model


def trainer_setup(config: DictConfig):
    trainer = L.Trainer(
        max_epochs=config.train.n_epochs,
        gpus=config.train.n_gpus,
        accelerator="auto",
        fast_dev_run=config.train.dev_run_n,
        log_every_n_steps=config.train.log_every_n_steps,
    )
    return trainer


def train(
    trainer: L.Trainer, data_module: PathContextDataModule, model: Code2Class | Code2Seq
):
    trainer.fit(model, datamodule=data_module)


def test(
    trainer: L.Trainer, data_module: PathContextDataModule, model: Code2Class | Code2Seq
):
    trainer.test(model, datamodule=data_module)


def predict(
    trainer: L.Trainer,
    data_module: PathContextDataModule,
    model: Code2Class | Code2Seq,
    embeddings_dump_file: str = None,
    predictions_dump_file: str = None,
):
    # result: list of shape n_batches * [(labels, output_logits, attention_weights, encoded_paths)]
    result = trainer.predict(model, datamodule=data_module)
    if not embeddings_dump_file:
        return
    decoder = Code2SeqResultsReader(model)
    # labels: [label_parts+1; batch_size] - +1 comes from <SOS> token starting each tensor
    # logits: [label_parts+1; batch_size; vocabulary_size]
    # attention_weights: [ label_parts+1; batch_size; max contexts per label in batch (changes between batches)]
    # paths: [total contexts in batch (changes between batches); embedding_size]
    predictions = ["label\tprediction"]
    f = open(embeddings_dump_file, "w", encoding="utf-8")
    columns = [f"x{i}" for i in range(-1,128)] #HACK: embedding size 
    columns[0] = "label"
    f.write("\t".join(columns))
    print("Writing results to file...")
    for labels, logits, attention_weights, paths, contexts_per_label in tqdm.tqdm(result):
        batch_write = []
        paths_split = paths.split(contexts_per_label.tolist(), dim=0)
        for label, logit, entry, path_embeddings, contexts_number in zip(labels.transpose(0,1), logits.transpose(0,1), attention_weights.transpose(0,1), paths_split, contexts_per_label):
            
            label = decoder.tensor2label(label)
            predicted_label = decoder.tensor2label(logit.argmax(1))
            predictions.append(f"\n{label}\t{predicted_label}")

            # First word is always <SOS> and so we skip the first row
            # We multiply path embeddings with attention weights of the entry
            # Lastly we combine all rows (there are label_parts-many) with sum
            # Thus we get an embedding size emgedding_size
            embedding = torch.matmul(entry[1:,0:contexts_number], path_embeddings).sum(0)

            columns[0] = label
            columns[1:] = [str(x.item()) for x in embedding]
            batch_write.append("\n" + "\t".join(columns))
        f.writelines(batch_write)
    f.close()

    if not predictions_dump_file:
        return

    with open(predictions_dump_file, "w", encoding="utf-8") as f:
        f.writelines(predictions)
            

def execute(mode: str, config: DictConfig):
    # Define data module
    data_module = PathContextDataModule(config.data_folder, config.data)
    # Define hyper parameters
    trainer = trainer_setup(config)

    mode = mode.split(",")

    if config.checkpoint:
        # Load model
        model = Code2Seq.load_from_checkpoint(config.checkpoint)
    else:
        # Define model
        model = model_setup(config, data_module)

    if "train" in mode:
        # Train model
        train(trainer, data_module, model)
    if "test" in mode:
        test(trainer, data_module, model)
    if "predict" in mode:
        predict(trainer, data_module, model, config.predict.embeddings_path, config.predict.compare_path)


if __name__ == "__main__":
    __arg_parser = argparse.ArgumentParser()
    __arg_parser.add_argument(
        "-config",
        "--config",
        dest="config",
        help="Path to YAML configuration file",
        type=str,
    )
    __arg_parser.add_argument(
        "-mode", "--mode", dest="mode", help="Set mode to train or test", type=str
    )
    if len(sys.argv) == 1:
        sys.argv += [
            "--config",
            "./code2seq-config.yaml",
            "--mode",
            "train,test",
        ]
    __args = __arg_parser.parse_args()

    __config = OmegaConf.load(__args.config)
    # print(OmegaConf.to_yaml(__config))
    execute(__args.mode, __config)
