from typing import List, Dict, Union

import gzip
import json
import pickle
import yaml
import os

import numpy as np
import jax.numpy as jnp

import logging

# A logger for this file
logger = logging.getLogger(__name__)


class NumpyEncoder(json.JSONEncoder):
    """Special json encoder for numpy types"""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, jnp.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def read_json(file_path: str):
    """Reads a json file, or its gzip-compressed copy `<file_path>.gz` when the plain file does not exist."""
    if not os.path.exists(file_path) and os.path.exists(file_path + ".gz"):
        with gzip.open(file_path + ".gz", "rt") as file:
            return json.load(file)
    with open(file_path, "r") as file:
        return json.load(file)


def read_yaml(file_path: str):
    """Reads a yaml file and returns a dictionary with its content."""
    with open(file_path, "r") as file:
        return yaml.safe_load(file)


def save_json(file_name: str, data: Dict[str, Union[List[float], np.ndarray]]):
    """Saves a list to a json file."""
    # Create directory if it does not exist
    if not os.path.exists(os.path.dirname(file_name)):
        os.makedirs(file_name, exist_ok=True)
    with open(file_name, "w", encoding="utf-8") as f:
        json.dump(data, f, cls=NumpyEncoder, indent=4)


def read_pickle(file_path: str):
    """Reads a pickle file and returns its content."""
    with open(file_path, "rb") as file:
        return pickle.load(file)


def save_pickle(file_path: str, data):
    """Writes a dictionary to a pickle file."""
    # Create the folder if it doesn't exist
    folder_path = os.path.dirname(file_path)
    if folder_path:
        os.makedirs(folder_path, exist_ok=True)

    with open(file_path, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def format_params(prior, result):
    """
    Helper function to format the learned parameters.
    """
    logger.info(f"NLL: {result['evidence']}, BIC: {result['bic']}, AIC: {result['aic']}")
    if prior == "exponential":
        lambda_, pi = result["lambda"], result["pi"]
        logger.info(f"λ: {lambda_}, π: {pi}")
        return {"lambda_": lambda_, "pi": pi}
    else:
        mu, sigma, pi = result["mu"], result["sigma"], result["pi"]
        logger.info(f"μ: {mu}, σ: {sigma}, π: {pi}")
        return {"logmu": mu, "std": sigma, "pi": pi}
