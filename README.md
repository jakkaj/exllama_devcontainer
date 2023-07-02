# Exllama wrappers



## exllama_devcontainer
A nice way to fire up exllama based models in a little api in docker. 

Basically it pulls [exllama code](https://github.com/turboderp/exllama) from github and then wraps it up in a little container. Exllama has docker support already, this just makes a new container that is a little api. 

You can run it in a sub container as an API or you can run it in the dev container directly. 

## Getting Started

Create `/data/models` on your local system, or edit the mount path in `./.devcontainer/devcontainer.json`. It's towards the bottom in the mounts section. 

```json
"mounts": [
		// map host ssh to container
		"source=${env:HOME}/.ssh,target=/home/vscode/.ssh,type=bind,consistency=cached",
		"source=/data,target=/data,type=bind,consistency=consistent"
	]
```

## Download a model

Grab the GTPQ models from Hugging face e.g something like [https://huggingface.co/TheBloke/Nous-Hermes-13B-GPTQ](https://huggingface.co/TheBloke/Nous-Hermes-13B-GPTQ) works well. 

```sh
git lfs install
git clone https://huggingface.co/TheBloke/Nous-Hermes-13B-GPTQ
```

## Note on GPT Requirement
This dev container requires a cuda capable GPU. 

### Run API in Docker

Edit `./docker/.env` and change the local path to your model. This will be loaded by the API. 

```
PORT=6010
RUN_UID=1000  # set to 0 to run the service as root inside the container
APPLICATION_STATE_PATH=/data  # path to the directory holding application state inside the container
MODEL_PATH=/data/models/TheBloke_Wizard-Vicuna-13B-Uncensored-SuperHOT-8K-GPTQ  # replace with the actual model path on the host
```

- Pull the repo
- Open in the dev container
- Run `make prepare`
- Run `make run_exllama_service_container`

## The API
The api is based on this [exllama sample](https://github.com/turboderp/exllama/blob/master/example_flask.py). It's been rejigged to be a class and uses waitress. 

It supports the same endpoints, `infer_precise`, `infer_creative` and `infer_sphinx`.

There is a file in `rest/test_api.rest` that will let you test the api from in VS Code once its up and running

## Run Locally

The dev container is built with everything you need to run locally.

- Pull the repo
- Open in the dev container
- Run `make prepare`

#### Langchain Examples

1. Open `langchain/langchain_sample.py`. Edit your model path around line 60 e.g. `model_path='/data/models/TheBloke_Nous-Hermes-13B-GPTQ'`. 
2. On the VS Code debug tab, select `Langchain Sample`. Press F5.



## Other Stuff...
Of course you can load the exists exllama samples, such as the web etc from this container without any extra work!

Run other make commands such as `build_docker_web` followed by `launch_docker_web` or `launch_text_bot`. Before doing this edit `.env` under the exllama folder. 

