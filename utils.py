"""
Author: Luigi Saetta
Date created: 2024-04-27
Date last modified: 2024-06-02
Python Version: 3.11
"""

import logging
import os
import toml

ENCODING = "utf-8"


def read_preamble(
    preamble_id: str,
    file_name="preamble_library.toml",
):
    """
    read a preamble
    preamble_id: the name of the preamble in the file
    """
    with open(file_name, "r", encoding=ENCODING) as file:
        preambles = toml.load(file)

    return preambles["cohere_preambles"][preamble_id]


def remove_path_from_ref(ref_pathname):
    """
    remove the path from source (ref)
    """
    ref = ref_pathname
    # check if / or \ is contained
    if len(ref_pathname.split(os.sep)) > 0:
        ref = ref_pathname.split(os.sep)[-1]

    return ref


def get_console_logger():
    """
    To get a logger to print on console
    """
    logger = logging.getLogger("ConsoleLogger")

    # to avoid duplication of logging
    if not logger.handlers:
        logger.setLevel(logging.INFO)

        handler = logging.StreamHandler()
        handler.setLevel(logging.INFO)

        formatter = logging.Formatter("%(levelname)s:\t  %(asctime)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    logger.propagate = False

    return logger


def format_docs(docs):
    """
    format docs for LCEL
    """
    return "\n\n".join(doc.page_content for doc in docs)


def read_configuration(file_path):
    """
    read the configuration from toml file
    """
    with open(file_path, "r", encoding="utf-8") as file:
        config = toml.load(file)

    return config


def print_configuration(config):
    """
    print the configuration used
    """
    logger = get_console_logger()

    preamble_id = config["oci"]["preamble_id"]
    preamble_used = read_preamble(preamble_id=preamble_id)

    logger.info("-----------------------------")
    logger.info("Configuration used:")
    logger.info(" Model ID: %s", config["oci"]["model_id"])
    logger.info(" Endpoint: %s", config["oci"]["endpoint"])
    logger.info(" Temperature: %s", config["llm"]["temperature"])
    logger.info(" Max tokens: %s", config["llm"]["max_tokens"])
    logger.info(" top_k: %s", config["llm"]["top_k"])
    logger.info(" top_p: %s", config["llm"]["top_p"])
    # identify the preamble used in preamble_library
    logger.info(" Preamble ID: %s", preamble_id)
    logger.info(" Preamble used:")
    logger.info(preamble_used)
    logger.info("")
    logger.info(" FastAPI port: %s", config["fastapi"]["api_port"])
    logger.info("")
    logger.info(" Verbose: %s", config["general"]["verbose"])
    logger.info("")
    logger.info("-----------------------------")

    logger.info("")
