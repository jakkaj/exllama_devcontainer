prepare:
	git submodule init
	git submodule update --remote

build_docker:
	cd exllama && docker-compose build
launch_docker:
	cd exllama && docker-compose up

launch_text_bot:
	docker exec -u user -it exllama-web-1 python3 example_chatbot.py -d /data/model -un "Jeff" -p prompt_chatbort.txt

run_benchmark:
	cd exllama && python test_benchmark_inference.py -d /data/models/TheBloke_Nous-Hermes-13B-SuperHOT-8K-GPTQ -p -ppl