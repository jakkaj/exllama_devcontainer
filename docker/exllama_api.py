import os
import glob
import dotenv
import json
from urllib.parse import parse_qs
from webob import Response
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
        self.app.route('/infer', methods=['POST'])(self.infer_context_params)
    
    

    def get_response(self, outputs, prompt):
        
        # # if ASSISTANT: is in the outputs then make outputs only the text after that
        # if "ASSISTANT:" in outputs:
        #     outputs = outputs.split("ASSISTANT:")[1]
        
        # for s in stop:
        #     print(f"Checking for stop sequence: {s}")
        #     if s in outputs:
        #         print(f"Found stop sequence: {s}")
        #         outputs = outputs.split(s)[0]                
            
        # # if prompt in outputs then remove it
        # if prompt in outputs:
        #     outputs = outputs.split(prompt)[1]
        
        response_dict = {
            "results": [{"text": outputs}]
        }
        response_json = json.dumps(response_dict)
        return response_dict
    
    def infer_context_params(self):
        query_string = request.query_string.decode('utf-8')
        query_params = parse_qs(query_string)
        
        data = request.json
        prompt = data.get('prompt')

        params = {
            'token_repetition_penalty_max': float(data.get('token_repetition_penalty_max', query_params.get('token_repetition_penalty_max', ['1.176'])[0])),
            'token_repetition_penalty_sustain': int(data.get('token_repetition_penalty_sustain', query_params.get('token_repetition_penalty_sustain', [str(self.config.max_seq_len)])[0])),
            'temperature': float(data.get('temperature', query_params.get('temperature', ['0.7'])[0])),
            'top_p': float(data.get('top_p', query_params.get('top_p', ['0.1'])[0])),
            'top_k': int(data.get('top_k', query_params.get('top_k', ['40'])[0])),
            'typical': float(data.get('typical', query_params.get('typical', ['0.0'])[0])),
            'max_new_tokens': int(data.get('max_new_tokens', query_params.get('max_new_tokens', ['200'])[0]))
        }
        
        # stop = data.get("stop_sequence")
        # if stop is not None:
        #     print(f"Stop sequence: {stop}")
        #     setattr(self.generator.settings, "stop_sequences", stop)
        
        # print("Parameters:")
        # for key, value in params.items():
        #     print(f"{key}: {value}")
        
        print(f"--------------\n{prompt}\n--------------")
            
        self.generator.settings.token_repetition_penalty_max = params['token_repetition_penalty_max']
        self.generator.settings.token_repetition_penalty_sustain = params['token_repetition_penalty_sustain']
        self.generator.settings.temperature = params['temperature']
        self.generator.settings.top_p = params['top_p']
        self.generator.settings.top_k = params['top_k']
        self.generator.settings.typical = params['typical']
        

        outputs = self.generator.generate_simple(prompt, max_new_tokens=params['max_new_tokens'])
        output_json = self.get_response(outputs, prompt)
        response = Response(json=output_json, content_type='application/json')
        return response

    
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
        output_json = self.get_response(outputs)
        response = Response(json=output_json, content_type='application/json')
        return response

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
        output_json = self.get_response(outputs)
        response = Response(json=output_json, content_type='application/json')
        return response

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
        output_json = self.get_response(outputs)
        response = Response(json=output_json, content_type='application/json')
        return response

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
