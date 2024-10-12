"""
This module creates an API on top of OCI Cohere
command-r/command-r-plus

It provides several operations:
    - answer
    - summarize

   V2 has been introduced to handle long transcriptions

    06/08/2024
        preparing for adding full multi-lingual to chat
        summarize: added support for he, nl, fr
    12/10/2024
        migrated to last version of langchain, langchain-community
        using ChatOCIGenai
"""

import traceback
from typing import List, Dict, Optional
import time

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel

from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_community.chat_models.oci_generative_ai import ChatOCIGenAI
from langchain_community.vectorstores import FAISS

from oci_cohere_embeddings_utils import OCIGenAIEmbeddingsWithBatch

from utils import (
    get_console_logger,
    print_configuration,
    read_configuration,
    read_preamble,
)
from utils_chuncking import split_in_chunks


# this represent the input to api
# the chat history is managed internally by API
# conv_id is passsed as parameter (?conv_id=)
class Message(BaseModel):
    """
    class for the body of the request

    query: the request from the user
    documents: list of documents to use to answer the query
    """

    query: str
    documents: List[str]


class MessageConfig(BaseModel):
    """ "
    The message to handle a change of configuration
    """

    token: str
    verbose: Optional[bool] = False
    preamble_id: Optional[str] = None
    # not using model_id cause conflicts with a Pydantic namespace
    id_model: Optional[str] = None


class MessageSummarize(BaseModel):
    """ "
    The message to handle summarization
    """

    language: Optional[str] = "en"
    """ language will be used to change the prompt"""
    documents: List[str]
    """The list of txt to summarize, normally 1"""


#
# Configs
#
# media type for the output
MEDIA_TYPE_NOSTREAM = "text/plain"
MEDIA_TYPE_NOSTREAM_JSON = "application/json"


#
# Main
#
app = FastAPI()

# global Object to handle conversation history
conversations: Dict[str, List[BaseMessage]] = {}

logger = get_console_logger()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app_config = read_configuration("config.toml")


#
# supporting functions to manage the conversation
# history (add, get)
#
def add_message(conv_id, role, txt):
    """
    add a msg to a conversation.
    If the conversation doesn't exist create it
    role: can be USER or CHAT
    txt: str, the text of the message
    """
    verbose = app_config["general"]["verbose"]

    if conv_id not in conversations:
        # create it
        if verbose:
            logger.info("Creating conversation id: %s", conv_id)

        conversations[conv_id] = []

    # add the request to the conversation
    if role == "USER":
        msg = HumanMessage(content=txt)
    else:
        # ai message
        msg = AIMessage(content=txt)

    # identify the conversation
    conversation = conversations[conv_id]
    # add the msg
    conversation.append(msg)

    if verbose:
        logger.info("Added msg to conversation id: %s", conv_id)

    # to keep only MAX_NUM_MSGS in the conversation
    if len(conversation) > app_config["llm"]["max_num_msgs"]:
        if verbose:
            logger.info("Removing old msg from conversation id: %s", conv_id)
        # remove first (older) el from conversation
        conversation.pop(0)


def get_conversation(v_conv_id):
    """
    return a conversation as List[CohereMessage]
    """
    if v_conv_id not in conversations:
        conversation = []
    else:
        conversation = conversations[v_conv_id]

    return conversation


def get_chat_model():
    """
    Build an instance of Chat Model
    """
    chat_model = ChatOCIGenAI(
        auth_type=app_config["oci"]["auth"],
        model_id=app_config["oci"]["model_id"],
        service_endpoint=app_config["oci"]["endpoint"],
        compartment_id=app_config["oci"]["compartment_ocid"],
        model_kwargs={
            "temperature": app_config["llm"]["temperature"],
            "max_tokens": app_config["llm"]["max_tokens"],
        },
    )
    return chat_model


def get_embedding_model():
    """
    Build an instance of Embedding Model
    """
    embed_model = OCIGenAIEmbeddingsWithBatch(
        auth_type=app_config["oci"]["auth"],
        model_id=app_config["embeddings"]["model_id"],
        service_endpoint=app_config["embeddings"]["embed_endpoint"],
        compartment_id=app_config["oci"]["compartment_ocid"],
    )
    return embed_model


#
# to handle chunking and semantic search (v2)
#
def handle_request_v2(request: Message, conv_id: str):
    """
    handle a request to LLM inside a conversation
    handle also chunking and semantic search into chunks
    conv_id : identify the conversation (chat_history)
    """
    # we could have input in more than 1 txt
    # split in chunks
    docs = split_in_chunks(request.documents)

    # create a Vector Store with Faiss
    embed_model = get_embedding_model()

    db = FAISS.from_documents(docs, embed_model)

    # do semantic search to retrieve a subset of chunks
    # results is a list of (doc, score), score=distance
    # default distance is L2, first returned are better
    results = db.similarity_search_with_score(
        request.query, k=app_config["retriever"]["k"]
    )

    # take only the txts
    docs_txt = [doc.page_content for (doc, score) in results]

    # prepare documents in the right format for Cohere
    # for now documents are provided from client as a List[str]

    # Cohere wants a map
    documents = [{"snippet": doc} for doc in docs_txt]

    # get the chat history from conv_id
    # if it is the first request it creates a new conversation
    # and you get []
    chat_history = get_conversation(conv_id)

    # create the client for OCI Cohere command-r/r-plus
    try:
        chat = get_chat_model()

        # here we invoke the model
        response = chat.invoke(
            request.query, chat_history=chat_history, documents=documents
        )
    except Exception as e:
        logger.error("Error in handle_request_v2:")
        logger.error(traceback.format_exc())
        logger.error(e)

    return response


def handle_summarize_v2(request: MessageSummarize):
    """
    handle a request to LLM to summarize a list of txt
    """
    full_content = "\n".join(request.documents)
    # added 09/07
    lang = request.language

    # need to be sure that the max lenght is not > context_window
    max_input_size = app_config["summarize"]["max_input_size"]

    if len(full_content) > max_input_size:
        logger.info("Truncating input for summarize...")
        full_content = full_content[:max_input_size]

    # Cohere wants a map
    documents = [{"snippet": full_content}]

    # create the client for OCI Cohere command-r/r-plus
    chat = get_chat_model()

    # handle language (09/07)
    request = read_preamble(f"request_sum_{lang}")

    # here we invoke the model
    # no chat_history
    response = chat.invoke(request, chat_history=[], documents=documents)

    return response


#
# HTTP handling methods
# Non streaming methods
#
tags_metadata = [
    {
        "name": "V1",
        "description": "V1",
    },
    {
        "name": "V2",
        "description": "Operation introduced to handle long transcriptions.",
    },
    {
        "name": "Configuration",
        "description": "Operation to handle config changes.",
    },
]


# to clean up a conversation
@app.delete("/delete/", tags=["V1"])
def delete(conv_id: str):
    """
    delete a conversation
    """
    logger.info("Called delete, conv_id: %s...", conv_id)

    if conv_id not in conversations:
        raise HTTPException(status_code=404, detail="Conversation not found")

    del conversations[conv_id]
    return {"conv_id": conv_id, "messages": []}


# to handle change in configs
@app.get("/get_config/", tags=["Configuration"])
def get_config():
    """
    return the current app configuration
    """
    return app_config


@app.post("/change_config/", tags=["Configuration"])
def change_config(request: MessageConfig):
    """
    handle the change of configuration

    saupported: verbose, preamble_id and model_id
    """
    if request.token == "4321":
        logger.info("Config change:")

        if request.verbose is not None:
            logger.info("New value for verbose: %s", request.verbose)
            app_config["general"]["verbose"] = request.verbose

        if request.preamble_id is not None:
            logger.info("New preamble id: %s", request.preamble_id)

            app_config["oci"]["preamble_id"] = request.preamble_id

            print_configuration(app_config)

        if request.id_model is not None:
            logger.info("New model id: %s", request.id_model)
    else:
        raise HTTPException(status_code=400, detail="Change not allowed.")


#
# V2 operations (handle long transcriptions)
#
@app.post("/v2/answer/", tags=["V2"])
def answer_v2(request: Message, conv_id: str):
    """
    Get a request + a set of documents and answer
    using command_r
    """
    time_start = time.time()

    logger.info("Called answer, conv_id: %s...", conv_id)

    try:
        response = handle_request_v2(request, conv_id)

        # extract only the text from response
        output = response.content

        if app_config["general"]["verbose"]:
            logger.info(response.content)

        # add request/response to conversation history
        add_message(conv_id, "USER", request.query)
        add_message(conv_id, "CHATBOT", output)

    except Exception as e:
        logger.error("Error in answer V2 %s", e)
        output = f"Error in answer V2: {e}"

    time_elapsed = time.time() - time_start
    logger.info("Elapsed time: %s sec.", round(time_elapsed, 1))
    logger.info("")

    return Response(content=output, media_type=MEDIA_TYPE_NOSTREAM)


@app.post("/v2/answer_with_citations/", tags=["V2"])
def answer_with_citation_v2(request: Message, conv_id: str):
    """
    Get a request + a set of documents and answer
    using command_r/r_plus
    """
    time_start = time.time()

    logger.info("Called answer_with_citations, conv_id: %s...", conv_id)

    try:
        response = handle_request_v2(request, conv_id)

        # extract the text and citations from response
        output = response.data

        if app_config["general"]["verbose"]:
            logger.info(response.data)

        # add request/response to conversation history
        add_message(conv_id, "USER", request.query)
        # only the txt is saved in the history
        add_message(conv_id, "CHATBOT", output.chat_response.text)

    except Exception as e:
        logger.error("Error in answer_with_citations V2 %s", e)
        output = f"Error in answer_with_citations V2: {e}"

    time_elapsed = time.time() - time_start
    logger.info("Elapsed time: %s sec.", round(time_elapsed, 1))
    logger.info("")

    return Response(content=str(output), media_type=MEDIA_TYPE_NOSTREAM_JSON)


@app.post("/v2/summarize/", tags=["V2"])
def summarize_v2(request: MessageSummarize):
    """
    a set of documents and summarize them
    """
    time_start = time.time()

    # for now only english, italian, spanish, hebrew
    logger.info("Called summarize, language: %s...", request.language)

    try:
        response = handle_summarize_v2(request)

        # extract only the text from response
        output = response.content

        if app_config["general"]["verbose"]:
            logger.info(output)

        # don't add request/response to conversation history

    except Exception as e:
        logger.error("Error in summarize V2 %s", e)
        output = f"Error in summarize V2: {e}"

    time_elapsed = time.time() - time_start
    logger.info("Elapsed time: %s sec.", round(time_elapsed, 1))
    logger.info("")

    return Response(content=output, media_type=MEDIA_TYPE_NOSTREAM)


# control ip and port
if __name__ == "__main__":
    print_configuration(app_config)

    uvicorn.run(host=app_config["fastapi"]["api_host"], port=app_config["fastapi"]["api_port"], app=app)
