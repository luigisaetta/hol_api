"""
Utils to handle chunking
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter

from utils import read_configuration, get_console_logger

app_config = read_configuration("config.toml")


def get_recursive_text_splitter():
    """
    return a recursive text splitter
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=app_config["splitting"]["max_chunk_size"],
        chunk_overlap=app_config["splitting"]["chunk_overlap"],
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter


def split_in_chunks(txts):
    """
    split input text in chunks
    txts: list of doc to split
    """
    logger = get_console_logger()

    text_splitter = get_recursive_text_splitter()

    docs = text_splitter.create_documents(txts)

    logger.info("splitted in %s chunks...", len(docs))

    return docs
