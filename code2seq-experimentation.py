import pytorch_lightning as L
import omegaconf
import argparse
import sys

from code2seq.model import Code2Class, Code2Seq
from code2seq.data.path_context_data_module import PathContextDataModule


def model_setup(config: omegaconf.DictConfig, data_module: PathContextDataModule):
    # model = Code2Class(  # Code2Class is apparently better for embedings
    #     config.model,
    #     config.optimizer,
    #     data_module.vocabulary,
    # )

    model = Code2Seq(
        config.model,
        config.optimizer,
        data_module.vocabulary,
        config.train.teacher_forcing,
    )

    return model


def trainer_setup(config: omegaconf.DictConfig):
    trainer = L.Trainer(
        max_epochs=config.train.n_epochs, gpus=config.train.n_gpus, accelerator="auto", fast_dev_run=config.train.dev_run_n
    )
    return trainer

def train(config: omegaconf.DictConfig):
    # Define data module
    data_module = PathContextDataModule(config.data_folder, config.data)

    # Define model
    model = model_setup(config, data_module)

    # Define hyper parameters
    trainer = trainer_setup(config)

    # Train model
    trainer.fit(model, datamodule=data_module)


def predict(model):
    pass


if __name__ == "__main__":
    __arg_parser = argparse.ArgumentParser()
    __arg_parser.add_argument(
        "config", help="Path to YAML configuration file", type=str
    )
    if len(sys.argv) == 1:
        sys.argv.append('d:\\Nik\\Projects\\mlfmf-poskusi\\code2seq-config.yaml')
    __args = __arg_parser.parse_args()

    __config = omegaconf.OmegaConf.load(__args.config)
    train(__config)