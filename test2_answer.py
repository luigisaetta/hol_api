"""
To test the API
"""

import time
from utils_tests import call_and_print_results, delete


# input for test

# this documents are used for RAG
# read the text from file
# Nome del file di testo
FILE_PATH = "document_test2.txt"

# Legge il contenuto del file
with open(FILE_PATH, "r", encoding="utf-8") as file:
    file_content = file.read()


documents = [file_content]

# conv_id will not change in following calls
params = {"conv_id": "2225"}

query_list = [
    # "Briefly summarize in no more than 5 sentences the text",
    "Is VisionCorp mentioned in the transcription?",
    "Make a list of the names of the people mentioned.",
    "Is Luigi mentioned in the document?",
    "Is Lisa mentioned in the document?",
    "Who is Lisa Miller?",
    "A person in Germany should see a policy valid in UAE?",
    "Where does Xander live?",
    "How many days of paternity leave is Xander eligible?",
    "How many days of annual leave is an italian employee allowed to take?",
    """Can Xander take 13 days more of annual leave?
    Organize in steps, compute remaining days of leave and then answer based on the result.""",
    # "What are the PII contained in the text?",
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
