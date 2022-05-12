import torch
import numpy as np
import pandas as pd
from model import Model
from config import Config
from PIL import Image
from torchvision import transforms
import os

def main():
    data = []
    for image_file in os.listdir(Config.input_dir):
        data.append([image_file, Image.open(f'{Config.input_dir}/{image_file}').convert('RGB')])
    
    testing_trasnforms = transforms.Compose(
        [transforms.Resize((64,128)),                    
        transforms.ToTensor(),                     
        transforms.Normalize(                      
        mean=[0.485, 0.456, 0.406],                
        std=[0.229, 0.224, 0.225])]
    )  
    classifier = Model(Config.model_name)
    classifier.load_state_dict(torch.load(f'{Config.exp_dir}/{Config.weights_file}')["state_dict"])
    classifier.eval()
    
    predictions, file_paths = [], []
    for image_path, image in data:
        image = torch.unsqueeze(testing_trasnforms(image), 0)
        y = np.argmax(classifier.forward(image)[0].detach().cpu().numpy())
        predictions.append(Config.label_format[y])
        file_paths.append(image_path)

    df = pd.DataFrame({'predictions':predictions, 'image': file_paths})
    df.to_csv(f'{Config.output_dir}/prediction.csv')

if __name__ == "__main__":
    main()