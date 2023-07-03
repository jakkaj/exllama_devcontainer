
import gzip
import pickle
import sys
import os
import dotenv
from llama_index import LangchainEmbedding, PromptHelper, download_loader, LLMPredictor, ServiceContext, GPTVectorStoreIndex, ResponseSynthesizer, StorageContext, GPTListIndex, load_index_from_storage
from langchain.chat_models import ChatOpenAI
from llama_index.indices.composability import ComposableGraph
from llama_index.indices.knowledge_graph.base import GPTKnowledgeGraphIndex
from llama_index.query_engine import RetrieverQueryEngine
from pathlib import Path
import markdown

import networkx as nx


import colored_traceback
import yaml

from integrations.refine_prompts import EXLLAMA_REFINE_PROMPT, EXLLAMA_TEXT_QA_PROMPT

from jorvis_tools.jorvis_context import JorvisContext
from jorvis_tools.jorvis_llm import JorvisLLM
colored_traceback.add_hook()


class DocumentProcessor:
    def __init__(self, jorvis_context):
        self.env_path = jorvis_context.env_path
        self.storage_dir = jorvis_context.storage_dir
        self.jorvis_context = jorvis_context
        self.load_env()

    def load_env(self):
        dotenv.load_dotenv(self.env_path)

    def index_and_store(self, service_context, source_dir, extension_filter=[".pdf"]):

        # if source_dir doesnt not exists, fail
        if not os.path.exists(source_dir):
            raise Exception(f"Source directory does not exist: {source_dir}")

        # check if data_dir exists
        if not os.path.exists(self.storage_dir):
            # create it
            os.makedirs(self.storage_dir)

        # process each directory
        return self._process_directory(
            service_context,  source_dir, extension_filter)

    def _load_docset_from_storage(self, persist_dir):
        # if storage_dir does not exist, fail
        if not os.path.exists(persist_dir):
            raise Exception(f"Storage directory does not exist: {persist_dir}")

        # if storage_file does not exist (doc_set.pickle), fail
        storage_file = f'{persist_dir}/doc.pickle.zip'
        if not os.path.exists(storage_file):
            raise Exception(f"Storage file does not exist: {storage_file}")
        # load the docset from the storage dir
        doc_set = []
        with open(storage_file, "rb") as f:
            # unzip and load the pickle
            with gzip.open(f, "rb") as g:
                doc = pickle.load(g)

        return doc

    def _save_docset(self, doc_set, persist_dir):
        # save the docset to the storage dir

        # ensure storage_dir exists
        if not os.path.exists(persist_dir):
            os.makedirs(persist_dir)

        storage_file = f'{persist_dir}/doc.pickle.zip'
        with open(storage_file, "wb") as f:
            with gzip.open(f, "wb") as g:
                pickle.dump(doc_set, g)

    def _process_directory(self, service_context, directory, extension_filter):
        UnstructuredReader = download_loader("UnstructuredReader")
        loader = UnstructuredReader()
        index_set = {}
        doc_set = {}
        # for each item in the directory
        for item in os.listdir(directory):
            path = os.path.join(directory, item)

            # if it's a sub-directory, process it recursively
            if os.path.isdir(path):
                update_result = self._process_directory(
                    service_context, path, extension_filter)
                index_set.update(update_result[0])
                doc_set.update(update_result[1])

            else:
                key = os.path.basename(path)

                if all(not key.endswith(ext) for ext in extension_filter):
                    continue

                # does persist dir exist? then continue
                persist_dir = f'{self.storage_dir}/{key}'

                _loaded_index = self._load_index(key, persist_dir)

                if _loaded_index is not None:
                    index_set[key] = _loaded_index
                    doc_set[key] = self._load_docset_from_storage(persist_dir)
                    continue

                print(f"Loading: {path}")
                doc_file = loader.load_data(file=Path(path))
                for d in doc_file:
                    d.extra_info = {"file": key}
                doc_set[key] = doc_file
                storage_context = StorageContext.from_defaults()
                cur_index = GPTListIndex.from_documents(
                    doc_file, service_context=service_context, storage_context=storage_context,
                    include_embeddings=True, include_metadata=True)
                index_set[key] = cur_index
                storage_context.persist(persist_dir=persist_dir)
                self._save_docset(doc_file, persist_dir)

        return (index_set, doc_set)

    def _load_index(self, key, persist_dir):

        if os.path.exists(persist_dir):
            print(f"Loading existing index: {key}")
            storage_context = StorageContext.from_defaults(
                persist_dir=persist_dir)
            loaded_index = load_index_from_storage(
                storage_context=storage_context, service_context=self.jorvis_context.service_context)
            return loaded_index

        return None

    def get_knowledge_graph(self, service_context, doc_set, key, max_triplets_per_chunk=5):
        persist_dir = f'{self.storage_dir}/{key}/knowledge_graph/{max_triplets_per_chunk}'
        _loaded_index = self._load_index(key, persist_dir)
        if _loaded_index is not None:
            return _loaded_index
        print(f"Generating Knowledge Graph: {key} ({max_triplets_per_chunk})")
        storage_context = StorageContext.from_defaults()

        kb = GPTKnowledgeGraphIndex.from_documents(doc_set[key],
                                                   include_embeddings=True,
                                                   service_context=service_context,
                                                   storage_context=storage_context,
                                                   max_triplets_per_chunk=max_triplets_per_chunk)
        storage_context.persist(persist_dir=persist_dir)
        return kb

    def query_regular(self, index, str_query, mode="compact", top_k=10):

        # "similarity_top_k": 4,
        #     "verbose": False,
        #     "mode": "embedding",
        #     "response_mode": "compact",
        #     "text_qa_template": QA_PROMPT,
        #     "refine_template": REFINE_PROMPT
        retriever = index.as_retriever(retriever_mode='default')
        response_synthesizer = ResponseSynthesizer.from_args(
            response_mode=mode, 
            service_context=self.jorvis_context.service_context,
            text_qa_template=EXLLAMA_TEXT_QA_PROMPT,
            refine_template=EXLLAMA_REFINE_PROMPT, 
            verbose=True)
        query_engine = RetrieverQueryEngine(
            retriever=retriever, response_synthesizer=response_synthesizer)

        # query_engine = index.as_query_engine(retriever_mode="default", response_mode=mode)

        response = query_engine.query(str_query)

        return response.response

    def load_meta(self, index_set):

        # if storage_dir does not exist, create it
        if not os.path.exists(self.storage_dir):
            os.makedirs(self.storage_dir)

        if os.path.exists(f"{self.storage_dir}/meta.yaml"):
            with open(f"{self.storage_dir}/meta.yaml", "r") as f:
                meta = yaml.load(f, Loader=yaml.FullLoader)
        else:
            meta = {}

        summarize_query = """Please provide a detailed summary of the document."""
        # iterate the keys in index_set
        for key in index_set.keys():
            if key in meta:
                continue
            print(f"Summarising: {key}")
            index = index_set[key]
            result = dp.query_regular(index, summarize_query, mode="refine")
            index.summary = result
            index.storage_context.persist()
            meta[key] = {"summary": result}
            print(result)

        # save meta to storage_dir yaml
        with open(f"{self.storage_dir}/meta.yaml", "w") as f:
            f.write(yaml.dump(meta))

        return meta

    # def to_mermaid(self, graph):
    #     mermaid_txt = 'graph TD;\n'
    #     for edge in graph.edges(data=True):
    #         # Add quotes around node names to handle spaces and include the edge title
    #         if 'title' in edge[2]:
    #             mermaid_txt += f'"{edge[0]}" -->|{edge[2]["title"]}| "{edge[1]}";\n'
    #         else:
    #             mermaid_txt += f'"{edge[0]}" --> "{edge[1]}";\n'

    #     # Add standalone nodes
    #     nodes_in_edges = set(edge[0] for edge in graph.edges(data=True)).union(
    #         set(edge[1] for edge in graph.edges(data=True))
    #     )
    #     for node in set(graph.nodes()) - nodes_in_edges:
    #         # Add quotes around node name to handle spaces
    #         mermaid_txt += f'"{node}"\n'

    #     return mermaid_txt


if __name__ == "__main__":



    jorvis_context = JorvisContext(
        storage_dir="/data/storage", env_path="./.env", is_exllama_llm=True, is_azure_llm=False, max_seq_len=4096, max_input_len=4096, compress_pos_emb=2)

    jorvis_llm = JorvisLLM(jorvis_context=jorvis_context)

    dp = DocumentProcessor(jorvis_context=jorvis_context)

    extension_filter = [".pdf", ".docx", ".txt", ".md"]

    llm = jorvis_llm.get_llm(temperature=.1, top_p=0)
    embeddings = LangchainEmbedding(jorvis_llm.get_embed_model())
    # llm=ChatOpenAI(
    #     temperature=.7, model_name="gpt-3.5-turbo", model_kwargs={"top_p": 1})
    
    
    
    llm_predictor = LLMPredictor(llm=llm)
    prompt_helper = PromptHelper(context_window=3000)
    
    service_context = ServiceContext.from_defaults(
        chunk_size_limit=256, llm_predictor=llm_predictor, embed_model=embeddings, prompt_helper=prompt_helper)

    jorvis_context.service_context = service_context

    index_set, doc_set = dp.index_and_store(service_context, "/data/source_documents",
                                            extension_filter=extension_filter)

    # list the keys in index_set
    # print(index_set.keys())
    meta = dp.load_meta(index_set)
    # exit()

    idx_query = index_set["2306.13643v1.pdf"]

    # kb_query = dp.get_knowledge_graph(service_context, doc_set, "Cloud_Native_Services_HLSD_30.docx")

    DEFAULT_TERM_STR = (
        """
Summarize the document.

"""
    )

    response = dp.query_regular(idx_query, DEFAULT_TERM_STR, mode="compact")

    # query_engine = kb_query.as_query_engine(
    #     include_text=True,
    #     response_mode="tree_summarize",
    #     embedding_mode='hybrid',
    #     similarity_top_k=5
    # )
    # response = query_engine.query(
    #     "Summarise the information in the document",
    # )
    print(response)

    # write the next output document to 

    # write outputs to ../data/output.md
    with open("./data/output.md", "w") as f:
        f.write(response)

    # g = kb_query.get_networkx_graph()
    # net = Network(notebook=True, cdn_resources="in_line", directed=True)
    # net.from_nx(g)
    # net.save_graph("network.html")

    # nodes = list(g.nodes)
    # edges = list(g.edges)

    # obj = {"nodes": nodes, "edges": edges}

    # #save obj as yaml
    # # if ../data/temp/ does not exist, create it
    # if not os.path.exists("../data/temp/"):
    #     os.makedirs("../data/temp/")

    # # nx.write_gml(g, "../data/temp/graph.gml")

    # # # open and read "../data/temp/graph.gml"
    # # with open("../data/temp/graph.gml", "r") as f:
    # #     gml_data = f.read()

    # # convert gml_data to mermaid code
    # mermaid_code = dp.to_mermaid(g)

    # # save mermaid_code as mermaid.html
    # with open("../data/temp/mermaid.txt", "w") as f:
    #     f.write(mermaid_code)

    # result = dp.query_and_display(idx_query, summarize_query)
    # print(result)
