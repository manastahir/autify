class Config():
    experiment_id = 'experiment_1'
    base_dir = '.'
    data_dir = f'./data'
    exp_dir = f'{base_dir}/experiments/{experiment_id}'
    
    label_format = {"checked": 0, "unchecked": 1, "other": 2}
    augmentations = True
    seed = 0
    splits = {"train": 0.8, "val": 0.1, "test": 0.1}
    freeze_until = "layer4"
    epochs = 20
    models = ["resnet50"]
    optimizers = ["Adam"]
    batch_sizes = [8]
    lr = [1e-3]