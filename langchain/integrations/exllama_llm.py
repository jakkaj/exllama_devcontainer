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


class ExLlamaApi(LLM):
    
    @property
    def _llm_type(self) -> str:
        return "custom"
        
    def _call(self, prompt: str, stop: Optional[List[str]]=None) -> str:        
        data = {
            'prompt': prompt,
            'max_new_tokens': 1500,            
            'temperature': .2,
            'top_p': 1,
            'top_k': 40,
            'typical': 0.0
        }

        # Add the stop sequences to the data if they are provided
        if stop is not None:
            data["stop_sequence"] = stop

        exllama_api_url = 'http://host.docker.internal:6010/infer'

        # Send a POST request to the Ooba API with the data
        response = requests.post(f'{exllama_api_url}', json=data)

        # Raise an exception if the request failed
        response.raise_for_status()

        # Check for the expected keys in the response JSON
        json_response = response.json()
        print(json_response['results'])
        if 'results' in json_response and len(json_response['results']) > 0 and 'text' in json_response['results'][0]:
            # Return the generated text
            text = json_response['results'][0]['text'].strip().replace("'''", "```")
            if text.startswith(prompt):
                text = text[len(prompt):].strip()
            # Remove the stop sequence from the end of the text, if it's there
            if stop is not None:
                for sequence in stop:
                    if sequence in text:
                        text = text.split(sequence)[0].rstrip()
                        if text.endswith(sequence):
                            text = text[: -len(sequence)].rstrip()

            print(text)
            return text
        else:
            raise ValueError('Unexpected response format from Ooba API')

    def __call__(self, prompt: str, stop: Optional[List[str]]=None) -> str:
        return self._call(prompt, stop)

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {}
