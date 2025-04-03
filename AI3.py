# Last amended: 26th March 2025
# Connected file:  'skill_gap_analysis_store_data.py'

# Objectives:
#           This file reads vector data from chroma vector 
#           store. It also connects to the web through Tavily.          
#           And then answers questions.

# API keys needed:
#           API keys are needed for LLM and for websearch
#           These need to be purchased.

# Run as:
#       streamlit run skill_gap_analysis_streamlit_app.py


#----------------- Libraries --------------

# 1.0
import streamlit as st
from llama_index.core.readers import SimpleDirectoryReader
from llama_index.readers.file import PagedCSVReader


# 1.1 The Settings is a bundle of commonly used resources used 
#     during the indexing and querying stage in a LlamaIndex workflow/application.
from llama_index.core import Settings


# 1.2 Ollama related
# https://docs.llamaindex.ai/en/stable/examples/embeddings/ollama_embedding/
from llama_index.embeddings.ollama import OllamaEmbedding


# 1.3 Vector store related
import chromadb
from llama_index.core import StorageContext
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.agent import FunctionCallingAgentWorker
from llama_index.core.agent import AgentRunner

# 1.4 Misc
import os
import pandas as pd



#--------BB. Model related--------------

# 2.0 Define embedding function
embed_model = OllamaEmbedding(
    model_name="nomic-embed-text",      # Using foundational model may be overkill
    base_url="http://localhost:11434",
    #dimensions=512,
    #ollama_additional_kwargs={"mirostat": 0},
)

Settings.embed_model = embed_model

from llama_index.llms.groq import Groq
# Following LLM gives good results
llm = Groq(
    model="qwen-qwq-32b",  # "llama-3.3-70b-versatile",
    api_key="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",
    temperature=0.5,
)

Settings.llm = llm

#llm.complete("What is AI").text


#------------------CC. Read Index from disk-----------
# 3.0 Load from disk

db2 = chromadb.PersistentClient(path="/home/ashok/Documents/chroma_db")
chroma_collection = db2.get_or_create_collection("datastore")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
index = VectorStoreIndex.from_vector_store(
    vector_store,
    embed_model=embed_model,
)


#-----------------DD. Tools -------------------

# 4.0  Tools

from llama_index.core.tools import QueryEngineTool

# 4.1 Query Engine
vector_query_engine = index.as_query_engine()

# 4.2 Query engine tool:

desc = "You will be given the area and city name, and your job is to find and give the properties in those areas or cities respectively from the rental2.csv file not from the web."
read_tool = QueryEngineTool.from_defaults(
    query_engine=vector_query_engine,
    # return_direct = True,
    description=(desc),
)

# 4.3 search jobs tool
from tavily import AsyncTavilyClient

async def social_life_on_web(query: str) -> str:
    """Given the City, Locality and Property, search for any nightlife, cafes, gyms, parks, cultural activities, active social communities, networking events, coworking spaces, also areas best suited for young professionals, families, or students. You are not to search for any other information on the web."""
    client = AsyncTavilyClient(api_key="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")
    return str(client.search(query))

async def review_on_web(query: str) -> str:
    """Given the City, Locality and Property, search for Croudsourced reviews from the web."""
    client = AsyncTavilyClient(api_key="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")
    return str(client.search(query))

async def nearby_area_on_web(query: str) -> str:
    """Given the City, Locality and Property, search on the web for nearby restaurants, hospitals, schools, grocery stores with ratings, community engagement events, weekend activities, and local festivals."""
    client = AsyncTavilyClient(api_key="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")
    return str(client.search(query))

# 4.5 Function tools from functions
from llama_index.core.tools import FunctionTool

# 4.5.1
social_tool = FunctionTool.from_defaults(fn=social_life_on_web)
review_tool = FunctionTool.from_defaults(fn=review_on_web)
nearby_tool = FunctionTool.from_defaults(fn=nearby_area_on_web)


#-------------GG. AgentRunner and AgentWorker------------

# 5.0 Define workers
agent_worker = FunctionCallingAgentWorker.from_tools(
    [read_tool, social_tool, review_tool, nearby_tool],
    llm=llm,
    verbose=True,  # Try also False
)

# 9.2 Define supervisor
agent = AgentRunner(agent_worker)


#-------------HH. Streamlit related------------



import streamlit as st

# Configure the Streamlit page
st.set_page_config(page_title="Chat with Rental Property Analyzer",
                   page_icon="💬🏠",
                   layout="wide")  # Wide layout for better spacing

# Apply custom CSS for styling
st.markdown(
    """
    <style>
    body {
        background-color: #f5f5f5;
    }
    .title {
        color: #2c3e50;
        text-align: center;
        font-weight: bold;
        font-size: 36px;
    }
    .subtitle {
        color: #34495e;
        text-align: center;
        font-size: 24px;
    }
    .chat-box {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1);
    }
    .container {
        background-color: #ecf0f1;
        padding: 15px;
        border-radius: 8px;
        text-align: center;
    }
    .sidebar {
        background-color: #dff9fb;
        padding: 10px;
        border-radius: 8px;
    }
    .chat-input {
        background-color: #fff;
        border: 1px solid #ddd;
        border-radius: 5px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Create a layout with three columns
col1, col2, col3 = st.columns([1, 1.75, 1])  # Adjust width ratios as needed

# Left column (Image inside a container)
with col1:
    with st.container():
        st.markdown('<div class="sidebar"><h3>🔹 Sample Prompts</h3></div>', unsafe_allow_html=True)
        st.image("/home/ashok/photo1.png", use_container_width=True)
        st.image("/home/ashok/photo3.png", use_container_width=True)

# Center column (Chat UI inside a container)
with col2:
    with st.container():
        st.markdown('<h1 class="title">💬🏠 Chat with Rental Property Analyzer</h1>', unsafe_allow_html=True)
        st.markdown('<p class="subtitle">Ask me anything about rental properties!</p>', unsafe_allow_html=True)

# Right column (Image inside a container)
with col3:
    with st.container():
        st.image("https://www.realtynmore.com/wp-content/uploads/2024/11/real-estate-market-in-bangalore-2024.jpg", use_container_width=True)
        st.image("https://www.crescent-builders.com/blog/wp-content/uploads/2021/07/Handshake-over-Property-Deal.original-e1619002308793.jpg", use_container_width=True)
        st.image("https://wp-assets.stessa.com/wp-content/uploads/2022/04/26080712/for-rent-sign-in-front-of-gray-house.jpg", use_container_width=True)

with col2:
    # Right side column (Content)
    # Title section with a custom color
  # st.title("Chat with Rental Property Analyzer 💬🏠")

    # Add some introductory text with color
#   st.markdown(
#       "<h3 style='color:#00796b;'>Welcome! Ask me anything about rental properties and their social life and nearby areas in Bangalore.</h3>",
#      unsafe_allow_html=True
#   )

    # 6.2 Initialize chat history if not present
    if "messages" not in st.session_state.keys():  
        st.session_state.messages = [
            {"role": "assistant", "content": "Ask me a question about rental properties!"}
        ]

    # 6.3 Initialize chat engine if not present
    if "chat_engine" not in st.session_state.keys():  
        st.session_state.chat_engine = agent

    # 6.4 Add a prompt for user input and save to chat history
    if prompt := st.chat_input("Ask a question about rental properties"):
        st.session_state.messages.append({"role": "user", "content": prompt})

    # 6.5 Display messages from the chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # 6.6 If the last message is not from the assistant, generate a new response
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            response_stream = st.session_state.chat_engine.chat(prompt)
            st.write(response_stream.response)
            message = {"role": "assistant", "content": response_stream.response}
            st.session_state.messages.append(message)
