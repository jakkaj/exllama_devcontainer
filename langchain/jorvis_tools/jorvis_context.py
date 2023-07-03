from dataclasses import dataclass

@dataclass
class JorvisContext:
    storage_dir: str
    env_path: str
    is_azure_llm: bool
    is_exllama_llm: bool
    
    
    max_seq_len: int = 2048
    max_input_len: int = 2048
    compress_pos_emb: int = 1
    
    # getter and setter for service_context 
    @property
    def service_context(self):
        return self._service_context
    
    @service_context.setter
    def service_context(self, service_context):
        self._service_context = service_context
        