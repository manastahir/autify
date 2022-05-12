docker volume create experiments
cd ./train
docker build -t trainautify --tag latest .
docker run -d -v experiments:/src/experiments --name training trainautify
docker container ls