"""
Utility functions for tests
Python: 3.11
"""

import json
import requests

from utils import read_configuration

# configs
app_config = read_configuration("config.toml")

API_PORT = app_config["fastapi"]["api_port"]

# cloud or local
TEST_ENV = "local"

if TEST_ENV == "cloud":
    MACHINE_IP = "130.61.183.159"
else:
    # local, when I test API on my Mac
    MACHINE_IP = "localhost"

# changed to V2 endpoint
URL_Q = f"http://{MACHINE_IP}:{API_PORT}/v2/answer"
URL_S = f"http://{MACHINE_IP}:{API_PORT}/v2/summarize"
URL_D = f"http://{MACHINE_IP}:{API_PORT}/delete"

# with gateway (07/08/2024)
# URL_Q = "https://bz6xe2jbqjezrrhilfgr6aru7e.apigateway.eu-frankfurt-1.oci.customer-oci.com/holv2/v2/answer"
# URL_S = "https://bz6xe2jbqjezrrhilfgr6aru7e.apigateway.eu-frankfurt-1.oci.customer-oci.com/holv2/v2/summarize"

headers = {"Content-type": "application/json"}


def call_and_print_results(query, v_params, v_documents):
    """
    helper function for the results
    """
    print("")
    print(f"Question: {query}")
    print("")

    # call the API
    data = {"query": query, "documents": v_documents}

    with requests.post(
        URL_Q, params=v_params, data=json.dumps(data), headers=headers, timeout=60
    ) as r:
        # handle the answer
        print(r.content.decode("utf-8"))

    print("")


def call_and_return_results(query, v_params, v_documents):
    """
    helper function for the results
    """
    # call the API
    data = {"query": query, "documents": v_documents}

    with requests.post(
        URL_Q, params=v_params, data=json.dumps(data), headers=headers, timeout=60
    ) as r:
        response_txt = r.content.decode("utf-8")

    return response_txt


def delete(v_params):
    """
    delete a conversation
    """
    requests.delete(URL_D, params=v_params, timeout=30)


def call_and_print_summarize(language, v_params, v_documents):
    """
    call summarize for test
    """
    data = {"language": language, "documents": v_documents}

    with requests.post(
        URL_S, params=v_params, data=json.dumps(data), headers=headers, timeout=60
    ) as r:
        print(r.content.decode("utf-8"))

    print("")
