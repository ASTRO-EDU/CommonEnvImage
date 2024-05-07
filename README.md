
=============
Docker image

docker system prune

-----

On MAC platform
docker build --platform linux/amd64 -t gamma:1.0.0 -f ./Dockerfile.amd .
docker build --platform linux/arm64 -t gamma:1.0.0 -f ./Dockerfile.arm .


On Linux platform
docker build -t gamma:1.0.0 -f ./Dockerfile.amd .

-----

docker build -t gamma:1.0.0 .

./bootstrap.sh gamma:1.0.0 $USER

docker run -it -d -v $HOME/workspace:/home/gamma/workspace -v /data02/:/data02/  -p 8001:8001 -p 8002:8002 --name rtadp1 gamma:1.0.0_$USER /bin/bash

docker exec -it rtadp1 /bin/bash
cd
. gamma.sh

nohup jupyter-lab --ip="*" --port 8001 --no-browser --autoreload --NotebookApp.token='gamma2024#'  --notebook-dir=/home/gamma/workspace --allow-root > jupyterlab_start.log 2>&1 &

OR

. gammapy.sh

nohup jupyter-lab --ip="*" --port 8002 --no-browser --autoreload --NotebookApp.token='gamma2024#'  --notebook-dir=/home/gamma/workspace --allow-root > jupyterlab_start.log 2>&1 &



