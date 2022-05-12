import torch

class DataGenerator(torch.utils.data.Dataset):
    def __init__(self, data, transforms = None):
        self.data = data
        self.transforms = transforms
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image, label = self.data[idx]
        if self.transforms:
            image = self.transforms(image)
        return image, torch.tensor(label, dtype=torch.long)