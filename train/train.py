import torch
import os, time, datetime, json
from PIL import Image
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from config import Config
from datagenerator import DataGenerator
from augmenters import training_trasnformations, testing_trasnformations
from model import Model
import utils

def main():
    os.makedirs(Config.exp_dir, exist_ok=True)
    data = {}
    for dir in os.listdir(Config.data_dir):
        if(not os.path.isdir(f'{Config.data_dir}/{dir}')):
            continue 
        data[dir] = []
        for image_file in os.listdir(f'{Config.data_dir}/{dir}'):
            data[dir].append([Image.open(f'{Config.data_dir}/{dir}/{image_file}').convert('RGB'), 
                              Config.label_format[dir]])
        np.random.seed(Config.seed)
        np.random.shuffle(data[dir])
    print(Config.exp_dir, os.listdir(Config.exp_dir))
    train, val, test = [], [], []
    for d in data:
        print(f'Total {len(data[d])} files in {d}')
        print(f'Training files in {d}: {int(Config.splits["train"]*len(data[d]))}')
        print(f'Training files in {d}: {int(Config.splits["val"]*len(data[d]))}')
        print(f'Training files in {d}: {int(Config.splits["test"]*len(data[d]))}')
        
        train.extend(data[d][:int(Config.splits["train"]*len(data[d]))])
        val.extend(data[d][int(Config.splits["train"]*len(data[d])): int((Config.splits["train"]+Config.splits["val"])*len(data[d]))])
        test.extend(data[d][int((Config.splits["train"]+Config.splits["val"])*len(data[d])):])

    print(len(train), len(val), len(test))

    counts = np.array([0]*len(Config.label_format))
    for i in train:
        counts[i[1]] +=1
    WEIGHTS = (np.sum(counts)-counts) / np.sum(counts)
    WEIGHTS 

    hyperparam_combinations = []
    utils.combine([Config.models]+[Config.optimizers]+[Config.lr]+[Config.batch_sizes], hyperparam_combinations)

    logger = {}
    for model_name, optimizer, lr, batch_size in hyperparam_combinations:
        print(f'RUN: {model_name}, {optimizer}, {lr}, {batch_size}')

        classifier = Model(model_name=model_name,
                          num_classes=len(Config.label_format), 
                          optimizer_name=optimizer,
                          lr=lr,
                          freeze_until="layer4"
                          )
        trainer = pl.Trainer(
                        # accelerator="cpu", 
                        # devices=1,
                        default_root_dir=Config.exp_dir,
                        deterministic=False, 
                        max_epochs=Config.epochs,
                        gradient_clip_val=1.0,
                        auto_lr_find=True,
                        callbacks=ModelCheckpoint(save_top_k=1,
                                                  monitor="valid_accuracy",
                                                  mode="max",
                                                  dirpath=Config.exp_dir,
                                                  filename=f"{model_name}-{lr}-{optimizer}-{batch_size}",
                                                  verbose=False,
                                                  save_weights_only=True)
                        )
        train_start = time.time()
        trainer.fit(classifier, 
                    torch.utils.data.DataLoader(DataGenerator(data=train, transforms=training_trasnformations), 
                                                batch_size=batch_size, 
                                                num_workers=1,
                                                shuffle = True), 
                    torch.utils.data.DataLoader(DataGenerator(data=val, transforms=testing_trasnformations), 
                                                batch_size=2, 
                                                num_workers=1, 
                                                shuffle=False))
        train_end = time.time()

        classifier.load_state_dict(torch.load(f'{Config.exp_dir}/{model_name}-{lr}-{optimizer}-{batch_size}.ckpt')["state_dict"])
        classifier.eval()
        Y, Yhat = [], []
        infer_start = time.time()
        for x,y in DataGenerator(data=test, transforms=testing_trasnformations):
            y_hat = torch.argmax(classifier.forward(torch.unsqueeze(x, 0))[0])
            Y.append(y.detach().cpu().numpy())
            Yhat.append(y_hat.detach().cpu().numpy())
        
        infer_end = time.time()
        accuracy = accuracy_score(Y, Yhat)
        f1 = f1_score(Y, Yhat, average='macro')
        precision = precision_score(Y, Yhat, average='macro')
        reacall = recall_score(Y, Yhat, average='macro')

        logger[f"run: {model_name}-{lr}-{optimizer}-{batch_size}"] = {
                                                    "train time": str(datetime.timedelta(seconds=train_end - train_start)),
                                                    "infer time": str(datetime.timedelta(seconds=infer_end - infer_start)),
                                                    "Test Accuracy": accuracy,
                                                    "Test F1": f1,
                                                    "Test Precision": precision,
                                                    "Test Recall": reacall}
    print(json.dumps(logger, indent=4))
    with open(f'{Config.exp_dir}/logging.json', 'w') as fp:
        json.dump(logger, fp, indent=4)
    print('Done!')

if __name__ == "__main__":
    main()