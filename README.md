
=============
Docker image

docker system prune

-----

On MAC platform
docker build --platform linux/amd64 -t gamma:1.0.1 -f ./Dockerfile.amd .
docker build --platform linux/arm64 -t gamma:1.0.1 -f ./Dockerfile.arm .


On Linux platform
docker build -t gamma:1.0.1 -f ./Dockerfile.amd .

-----

docker build -t gamma:1.0.1 .

./bootstrap.sh gamma:1.0.1 $USER

docker run -it -d -v $HOME/workspace:/home/gamma/workspace -v /data02/:/data02/  -p 8001:8001 -p 8002:8002 -p 8003:8003 --name gammasim gamma:1.0.0_$USER /bin/bash

docker exec -it gammasim /bin/bash
cd
. gamma.sh

nohup jupyter-lab --ip="*" --port 8001 --no-browser --autoreload --NotebookApp.token='gamma2024#'  --notebook-dir=/home/gamma/workspace --allow-root > jupyterlab_start.log 2>&1 &

OR

. gammapy.sh

nohup jupyter-lab --ip="*" --port 8002 --no-browser --autoreload --NotebookApp.token='gamma2024#'  --notebook-dir=/home/gamma/workspace --allow-root > jupyterlab_start.log 2>&1 &



docker run --gpus all -it -d -v $HOME:/home/gamma/workspace -v /data02/:/data02/ -p 8003:8003 --name gammasky_dl gammasky_dl:v0.4_$USER /bin/bash

docker exec -it gammasky_dl /bin/bash

nohup jupyter-lab --ip="*" --port 8003 --no-browser --autoreload --NotebookApp.token='gamma2024#' --notebook-dir=/home/gamma/ --allow-root > jupyterlab_start.log 2>&1 &
