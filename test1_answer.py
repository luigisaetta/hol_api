"""
To test the API
"""

import time

from utils_tests import call_and_print_results, delete


# input for test

# this documents are used for RAG
documents = [
    """Oracle database services and products offer customers cost-optimized and 
        high-performance versions of Oracle Database, 
        the world's leading converged, multi-model database management system, as well as in-memory, 
        NoSQL and MySQL databases. Oracle Autonomous Database, 
        available on premises via Oracle Cloud@Customer or in the Oracle Cloud Infrastructure, 
        enables customers to simplify relational database environments 
        and reduce management workloads.""",
    """Oracle AI services are a collection of offerings, including Generative AI, 
        with prebuilt AI models that make it easier for developers to apply AI 
        to applications and business operations. The models can be custom trained 
        for more accurate business results. Teams within an organization can reuse the models, 
        data sets, and data labels across services. 
        The services let developers easily add AI and machine learning to apps without slowing 
        application development.""",
    """Luigi Saetta was born in Naples, in 1965. He studied Physics at Federico II University.
       He lives in Rome, since 1992.
       He has joined Oracle in 1996 (age ago).
       Now he is member of Emea AI Specialist Team, as Generative AI Specialist.
    """,
    """Luigi Saetta has graduated in Physics in 1991.
       His thesis was about non-linear optic of Liquid Crystal.
    """
    """He is member of Emea AI Specialist Team.
       The team is leaded by A. Negrea.
       Members of the team are: M. de Grunt, A. Panda, C. Lemnaru, H. Jain, L. Saetta, S. Varghese
       A. Stellatos, Jesus B. Jimenez, M. Dikens and other people. 
    """,
    """
    In his free time Luigi enjoys to practice running and triathlon. 
    """,
]

# conv_id wil not change in following calls
params = {"conv_id": "2224"}

query_list = [
    "Tell me about Luigi Saetta",
    "Where does he live?",
    "What has he studied at University?",
    "Who are the members of the team he belongs to?",
    "Is Luigi married?",
    "Regarding sports, is he a sailor?",
    "Who is Lisa?",
    "Who is Lisa Miller?",
    "What are the names of Luigi's daughters?",
    "What are Oracle AI services?",
]

# do all the queries and print the response
for query in query_list:
    # to avoid too many calls in 1 minute
    time.sleep(1)

    call_and_print_results(query, params, documents)

# clean, delete the conversation
delete(params)
