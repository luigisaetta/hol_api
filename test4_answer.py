"""
To test the API
"""

import time
from utils_tests import call_and_print_results, delete


# input for test

# this documents are used for RAG
# read the text from file
# Nome del file di testo
FILE_PATH = "document_test4.txt"

# Legge il contenuto del file
with open(FILE_PATH, "r", encoding="utf-8") as file:
    file_content = file.read()


documents = [file_content]

# conv_id will not change in following calls
params = {"conv_id": "4001"}

query_list = [
    "Quali sono le industries menzionate nel testo?",
    # "Fai un elenco dei casi d'uso per il settore Manufacturing?",
    "Fai un elenco dei casi d'uso per il settore Fashion?",
    "Fai un elenco dei casi d'uso per il settore Aereonatico?",
]

print("")

# do all the queries and print the response
for i, query in enumerate(query_list):
    # to avoid too many calls in 1 minute
    time.sleep(5)

    print("---> Test. n.: ", i + 1)

    call_and_print_results(query, params, documents)

# clean, delete the conversation
delete(params)
