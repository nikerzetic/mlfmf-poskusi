import pytorch_lightning as L
import argparse
import sys

from typing import *
from omegaconf import OmegaConf, DictConfig
from code2seq.model import Code2Class, Code2Seq
from code2seq.data.path_context_data_module import PathContextDataModule


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
    save_to: str = None,
):
    # n_batches * [(labels, output_logits, attention_weights, encoded_paths)]
    result = trainer.predict(model, datamodule=data_module)
    # model.id2label
    if not save_to:
        return
    with open(save_to, "w", encoding="utf-8") as f:
        pass


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
        # Predict model
        model = Code2Seq.load_from_checkpoint(config.checkpoint)
        predict(trainer, data_module, model, config.predict.file_path)


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
            "d:\\Nik\\Projects\\mlfmf-poskusi\\code2seq-config.yaml",
            "--mode",
            "predict",
        ]
    __args = __arg_parser.parse_args()

    __config = OmegaConf.load(__args.config)
    # print(OmegaConf.to_yaml(__config))
    execute(__args.mode, __config)
