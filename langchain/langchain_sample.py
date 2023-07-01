#Remember to run the docker stuff first!
import json
import os
import re
import requests
import langchain
from langchain.chains import ConversationChain, LLMChain, LLMMathChain, TransformChain, SequentialChain
from langchain.chat_models import ChatOpenAI
from langchain.docstore import InMemoryDocstore
from langchain.llms.base import LLM, Optional, List, Mapping, Any
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.memory import (
    ChatMessageHistory,
    ConversationBufferMemory,
    ConversationBufferWindowMemory,
    ConversationSummaryBufferMemory
)
from langchain.prompts.prompt import PromptTemplate
from langchain.schema import messages_from_dict, messages_to_dict
from langchain.vectorstores import Chroma
from langchain.agents import load_tools
from langchain.agents import initialize_agent

from integrations.exllama_llm import ExLlamaApi


llm = ExLlamaApi()

tools = load_tools(["wikipedia", "llm-math"], llm=llm)

agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)

#In what year was the film Departed with Leopnardo Dicaprio released? What is this year raised to the 0.43 power?
agent.run("### Instruction:\nIn what year was the film Departed with Leopnardo Dicaprio released? What is this year raised to the 0.43 power?\n### Response:\n")