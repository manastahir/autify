This drive folder contains the logs, training results and weights of the experiment that I ran using experiments.ipynb file on colab. You can replicate the experiment by either running the notebook or by running the train.sh which create a docker container and runs the training inside it. https://drive.google.com/drive/folders/1-IiQ__GbyH5fu3wymXT1hLmBHgmLbozm?usp=sharing

run inference using weights downloaded from drive
1. set the name and weight's file_name in the ./infer/config.py 
2. place the images in the ./infer/input folder
3. adjust the global paths to the input and output folder in infer.sh
4. use bind mount instead of volume in ./infer.sh to mount the folder containing the weights 

To run the training.
1. set the seed value in the ./train/config.py to 0
2. run train.sh

To run the inference
1. set the name and weight's file_name in the ./infer/config.py 
2. place the images in the ./infer/input folder
3. adjust the global paths to the input and output folder in infer.sh
4. run ./infer.sh

Note:
The inference container uses bind mounts for input and output directories and the global paths for the input and output directories need to set according to your machine.

You can run docker volume inspect experiments to check the global path of the volume used by the training container, it contains the training logs and model weights. If istalled, you can use run the tensorboard to see the training logs.
