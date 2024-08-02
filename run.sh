#!/bin/bash

docker build -t qrl $(pwd)

if [ "$(docker ps -aq -f name=$CONTAINER_NAME)" ]; then
    docker rm -f qrl_container
fi

docker run -d -p 8888:8888 -v $(pwd):/workspace --name qrl_container qrl
sleep 5 && python -m webbrowser "http://localhost:8888?token=qrl"