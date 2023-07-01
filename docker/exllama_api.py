import os
import glob
import dotenv
from model import ExLlama, ExLlamaCache, ExLlamaConfig
from flask import Flask, request
from tokenizer import ExLlamaTokenizer
from generator import ExLlamaGenerator
from waitress import serve

class ExLlamaService:
    def __init__(self, model_directory):
        self.model_directory = model_directory
        self.app = Flask(__name__)

    def setup(self):
        tokenizer_path = os.path.join(self.model_directory, "tokenizer.model")
        model_config_path = os.path.join(self.model_directory, "config.json")
        st_pattern = os.path.join(self.model_directory, "*.safetensors")
        model_path = glob.glob(st_pattern)[0]

        self.config = ExLlamaConfig(model_config_path)
        self.config.model_path = model_path

        self.model = ExLlama(self.config)
        print(f"Model loaded: {model_path}")

        tokenizer = ExLlamaTokenizer(tokenizer_path)
        cache = ExLlamaCache(self.model)
        self.generator = ExLlamaGenerator(self.model, tokenizer, cache)

        # Set up routes
        self.app.route('/infer_precise', methods=['POST'])(self.infer_context_p)
        self.app.route('/infer_creative', methods=['POST'])(self.infer_context_c)
        self.app.route('/infer_sphinx', methods=['POST'])(self.infer_context_s)

    def infer_context_p(self):
        data = request.json
        prompt = data.get('prompt')

        self.generator.settings.token_repetition_penalty_max = 1.176
        self.generator.settings.token_repetition_penalty_sustain = self.config.max_seq_len
        self.generator.settings.temperature = 0.7
        self.generator.settings.top_p = 0.1
        self.generator.settings.top_k = 40
        self.generator.settings.typical = 0.0

        outputs = self.generator.generate_simple(prompt, max_new_tokens=200)
        return outputs

    def infer_context_c(self):
        data = request.json
        prompt = data.get('prompt')

        self.generator.settings.token_repetition_penalty_max = 1.1
        self.generator.settings.token_repetition_penalty_sustain = self.config.max_seq_len
        self.generator.settings.temperature = 0.72
        self.generator.settings.top_p = 0.73
        self.generator.settings.top_k = 0
        self.generator.settings.typical = 0.0

        outputs = self.generator.generate_simple(prompt, max_new_tokens=200)
        return outputs

    def infer_context_s(self):
        data = request.json
        prompt = data.get('prompt')

        self.generator.settings.token_repetition_penalty_max = 1.15
        self.generator.settings.token_repetition_penalty_sustain = self.config.max_seq_len
        self.generator.settings.temperature = 1.99
        self.generator.settings.top_p = 0.18
        self.generator.settings.top_k = 30
        self.generator.settings.typical = 0.0

        outputs = self.generator.generate_simple(prompt, max_new_tokens=200)
        return outputs

    def start(self, host="0.0.0.0", port=8004):
        print(f"Starting server on address {host}:{port}")
        
        
        serve(self.app, host=host, port=port)

if __name__ == '__main__':
    model_directory = "/data/model"
    dotenv.load_dotenv()
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8004))
    service = ExLlamaService(model_directory)
    service.setup()
    service.start(host=host, port=port)
