from torchvision import transforms

training_trasnformations = transforms.Compose(
    [transforms.Resize((64,128)),
    #  transforms.TrivialAugmentWide(),
     transforms.RandomVerticalFlip(p=0.5),
     transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),                    
     transforms.ToTensor(),                     
     transforms.Normalize(                      
     mean=[0.485, 0.456, 0.406],                
     std=[0.229, 0.224, 0.225])]
)

testing_trasnformations = transforms.Compose(
    [transforms.Resize((64,128)),                    
     transforms.ToTensor(),                     
     transforms.Normalize(                      
     mean=[0.485, 0.456, 0.406],                
     std=[0.229, 0.224, 0.225])]
)  