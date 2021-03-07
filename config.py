import argparse
import os

import torch
import yaml


def load_config():
    parser = argparse.ArgumentParser(description="fusion_training")
    parser.add_argument(
        "--config", type=str, help="Path to the YAML config file", required=True
    )
    args = parser.parse_args()
    config = _load_config_yaml(args.config)
    return config


def _load_config_yaml(config_file):
    return yaml.safe_load(open(config_file, "r"))
