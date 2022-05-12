class Config():
    experiment_id = 'experiment_1'
    base_dir = '.'
    input_dir = f'./input'
    output_dir = f'./output'
    exp_dir = f'{base_dir}/experiments/{experiment_id}'
    
    model_name = "resnet50"
    weights_file = "resnet50-0.001-Adam-8.ckpt"
    label_format = {0:"checked", 1:"unchecked", 2:"other"}
