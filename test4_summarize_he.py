"""
To test the API
"""

from utils_tests import call_and_print_summarize


# input for test

# this documents are used for RAG
# read the text from file
# Nome del file di testo
FILE_PATH = "document_test3.txt"

# Legge il contenuto del file
with open(FILE_PATH, "r", encoding="utf-8") as file:
    file_content = file.read()


documents = [file_content]

# conv_id will not change in following calls
params = {"conv_id": "0009"}

LANGUAGE = "he"
call_and_print_summarize(LANGUAGE, params, documents)
