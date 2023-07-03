import os
import dotenv
import openai
from langchain.chat_models import ChatOpenAI
from langchain.llms.openai import AzureOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from integrations.langchain_llm import Exllama, BasicStreamingHandler
from langchain.embeddings import HuggingFaceEmbeddings, SentenceTransformerEmbeddings

class JorvisLLM:
    def __init__(self, jorvis_context):
        self.jorvis_context=jorvis_context
        

    def _prep_env(self):
        dotenv.load_dotenv(self.jorvis_context.env_path)
        if self.jorvis_context.is_azure_llm is True:
            OPENAI_API_KEY_AZURE = os.environ.get("OPENAI_API_KEY_AZURE")    
            os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY_AZURE       
                         
            AZURE_OPENAI_SERVICE = os.environ.get("AZURE_OPENAI_SERVICE") or "myopenai"
                 
            openai.api_type = "azure"
            openai.api_base = f"https://{AZURE_OPENAI_SERVICE}.openai.azure.com"
            openai.api_version = "2023-05-15"
            openai.api_key = OPENAI_API_KEY_AZURE  
    
    def get_embed_model(self):
        
        if self.jorvis_context.is_exllama_llm is True:
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            return embeddings
        
        if self.jorvis_context.is_azure_llm is True:
            ADA_DEPLOYMENT_NAME = os.environ.get("ADA_DEPLOYMENT_NAME")        
            embeddings = OpenAIEmbeddings(
                model=ADA_DEPLOYMENT_NAME, deployment=ADA_DEPLOYMENT_NAME, chunk_size=1)
            return embeddings
        
        embeddings = OpenAIEmbeddings()
        return embeddings

    
    def get_llm(self, temperature=0.0, top_p=1):
        if self.jorvis_context.is_exllama_llm is True:
            return self.get_exllama_llm(temperature=temperature, top_p=top_p)
        else:
            return self.get_openai_llm(temperature=temperature, top_p=top_p)
    
    def get_exllama_llm(self, temperature=0.0, top_p=1):
        self._prep_env()
        model_path = os.environ.get("MODEL_PATH")
        handler = BasicStreamingHandler()
        llm = Exllama(streaming = True,
                #model_path='/data/models/TheBloke_WizardLM-33B-V1.0-Uncensored-SuperHOT-8K-GPTQ', 
                #model_path='/data/models/TheBloke_Wizard-Vicuna-13B-Uncensored-SuperHOT-8K-GPTQ', 
                #model_path='/data/models/TheBloke_Wizard-Vicuna-30B-Uncensored-GPTQ',
                model_path=model_path,
                lora_path = None,
                temperature = temperature,
                top_p = top_p,
                beams = 0, 
                beam_length = 0, 
                stop_sequences=["Human:", "User:", "AI:", "Observation:"],
                callbacks=[handler],
                verbose = False,
                max_seq_len = self.jorvis_context.max_seq_len,
                compress_pos_emb=self.jorvis_context.compress_pos_emb,
                max_input_len=self.jorvis_context.max_input_len
                #alpha_value = 4.0, #For use with any models
                #compress_pos_emb = 2, #For use with superhot
                #set_auto_map = "3, 2" #Gpu split, this will split 3gigs/2gigs.
              
              )
        handler.set_llm(llm)
        return llm
    


    def get_openai_llm(self, temperature=0.0, top_p=1):
        self._prep_env()
        if self.jorvis_context.is_azure_llm is True:
            AZURE_OPENAI_CHATGPT_DEPLOYMENT = os.environ.get("AZURE_OPENAI_CHATGPT_DEPLOYMENT")
                 
            llm = AzureOpenAI(deployment_name=AZURE_OPENAI_CHATGPT_DEPLOYMENT,
                temperature=temperature, top_p = top_p, model_name="gpt-35-turbo")
            return llm
        
        llm = ChatOpenAI(
            temperature=temperature, model_name="gpt-3.5-turbo", model_kwargs={"top_p": top_p})
        
        return llm