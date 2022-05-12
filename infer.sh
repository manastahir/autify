cd ./infer
docker build -t inferautify --tag latest .
docker run -v experiments:/src/experiments \
-v /home/niche-6/Documents/Anas/Misc/sandbox/infer/input:/src/input \
-v /home/niche-6/Documents/Anas/Misc/sandbox/infer/output:/src/output \
inferautify