import os
import sys
import logging
from os.path import join, dirname
from omegaconf import OmegaConf, ListConfig, DictConfig
from utils.distributed import init_distributed_mode, is_main_process
from utils.logger import setup_logger


logger = logging.getLogger(__name__)


def convert_types(config):
    """Convert `'None'` (str) --> `None` (None). Only supports top-level"""
    for k, v in config.items():
        if isinstance(v, DictConfig):
            setattr(config, k, convert_types(v))

        # TODO convert types in ListConfig, right now they are ignored
        # if isinstance(v, ListConfig):
        #     new_v = ListConfig()

        if v in ["None", "none"]:
            setattr(config, k, None)
    return config


def setup_config():
    """Conbine yaml config and command line config with OmegaConf.
    Also converts types, e.g., `'None'` (str) --> `None` (None)
    """
    config_path = sys.argv[1]
    del sys.argv[1]  # not needed
    cli_args = sys.argv[1:]
    yaml_config = OmegaConf.load(config_path)
    cli_config = OmegaConf.from_cli() if len(cli_args) else OmegaConf.create()
    # the latter overwrite the former, i.e, cli_config higher priority.
    logger.info(f"Command line configs: {cli_config}")
    config = OmegaConf.merge(yaml_config, cli_config)
    config = convert_types(config)
    if config.debug:
        config.wandb.enable = False
    return config


def setup_evaluate_config(config):
    """setup evaluation default settings, e.g., disable wandb"""
    assert config.evaluate
    config.wandb.enable = False
    if config.output_dir is None:
        config.output_dir = join(dirname(config.pretrained_path), "eval")
    return config


def setup_output_dir(output_dir, excludes=["code"]):
    """ensure not overwritting an exisiting/non-empty output dir"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=False)
    else:
        existing_dirs_files = os.listdir(output_dir)  # list
        remaining = set(existing_dirs_files) - set(excludes)
        remaining = [e for e in remaining if "slurm" not in e]
        assert len(remaining) == 0, f"remaining dirs or files: {remaining}"


def setup_main():
    """
    Setup config, logger, output_dir, etc. 
    Shared for pretrain and all downstream tasks.
    """
    config = setup_config()
    if hasattr(config, "evaluate") and config.evaluate:
        config = setup_evaluate_config(config)    
    init_distributed_mode(config)

    if is_main_process():
        setup_output_dir(config.output_dir, excludes=["code"])
        setup_logger(output=config.output_dir, color=True, name="loopitr")
        OmegaConf.save(
            config, open(os.path.join(config.output_dir, 'config.yaml'), 'w'))
    return config
