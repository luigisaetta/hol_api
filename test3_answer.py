"""
To test the API
"""

import time
from utils_tests import call_and_print_results, delete


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
params = {"conv_id": "2226"}

query_list = [
    "Make a list of the names of people in the text",
    "Who is Logan Giles?",
    "Is Pratap mentioned in the text?",
    "What Pratap mentioned in the text?",
    "What can you tell me about Named Entity Recognition?",
    "What about Sentiment Analysis?",
    "How many characters the model mask for sensible data?",
    "Summarize all the text",
]

print("")

# do all the queries and print the response
for i, query in enumerate(query_list):
    # to avoid too many calls in 1 minute
    time.sleep(1)

    print("---> Test. n.: ", i + 1)

    call_and_print_results(query, params, documents)

# clean, delete the conversation
delete(params)
