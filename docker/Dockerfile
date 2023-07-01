FROM exllama-web:build

COPY ./docker_entrypoint.sh /app/docker_entrypoint.sh
COPY ./exllama_api.py /app/exllama_api.py
COPY ./.env /app/.env

COPY requirements.txt /tmp/pip-tmp/

RUN python3 -m pip --disable-pip-version-check --no-cache-dir install -r /tmp/pip-tmp/requirements.txt \
    && rm -rf /tmp/pip-tmp

ENTRYPOINT ["/bin/bash", "-c", "/app/docker_entrypoint.sh"]
