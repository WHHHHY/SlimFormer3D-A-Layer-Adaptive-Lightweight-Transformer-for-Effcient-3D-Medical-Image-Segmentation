import argparse
import os
import nni
import torch
from lib import utils, dataloaders, models, losses, metrics, trainers

params_BTCV = {
    # ——————————————————————————————————————————————    Launch Initialization    —————————————————————————————————————————————————
    "CUDA_VISIBLE_DEVICES": "0,1,2,3",
    "seed": 1777777,
    "cuda": True,
    "benchmark": False,
    "deterministic": True,
    # —————————————————————————————————————————————     Preprocessing      ————————————————————————————————————————————————————
    "resample_spacing": [ 1.5,
        1.5,
        1.5],
    "clip_lower_bound": -200,
    "clip_upper_bound": 500,
    "samples_train": 1024,
    "samples_valid": 340,
    "crop_size": (96, 96, 96),
    "crop_threshold": 0.1,
    # ——————————————————————————————————————————————    Data Augmentation    ——————————————————————————————————————————————————————
    "augmentation_probability": 0.5,
    "augmentation_method": "Choice",
    "open_elastic_transform": True,
    "elastic_transform_sigma": 15,
    "elastic_transform_alpha": 1,
    "open_gaussian_noise": True,
    "gaussian_noise_mean": 0,
    "gaussian_noise_std": 0.01,
    "open_random_flip": True,
    "open_random_rescale": True,
    "random_rescale_min_percentage": 0.5,
    "random_rescale_max_percentage": 1.5,
    "open_random_rotate": True,
    "random_rotate_min_angle": -50,
    "random_rotate_max_angle": 50,
    "open_random_shift": True,
    "random_shift_max_percentage": 0.3,
    "normalize_mean": 0.456  ,
    "normalize_std": 0.224,
    # —————————————————————————————————————————————    Data Loading     ——————————————————————————————————————————————————————
    "dataset_name": "BTCV",
    "dataset_path": r"./datasets/BTCV",
    "create_data": False,
    "batch_size": 18,
    "num_workers": 30,
    # —————————————————————————————————————————————    Model     ——————————————————————————————————————————————————————
    "model_name": "SlimFormer",
    "in_channels": 1,
    "classes": 14,
    "scaling_version": "TINY",
    "dimension": "3d",
    "index_to_class_dict" : {
        0: "background",
        1: "spleen",
        2: "rkid",
        3: "lkid",
        4: "gall",
        5: "eso",
        6: "liver",
        7: "sto",
        8: "aorta",
        9: "IVC",
        10: "veins",
        11: "pancreas",
        12: "rad",
        13: "lad"
    },

    "resume": None,
    "pretrain": None,
    # ——————————————————————————————————————————————    Optimizer     ——————————————————————————————————————————————————————
    "optimizer_name": "Adam",
    "learning_rate": 0.0005,
    "weight_decay": 0.00005,
    "momentum": 0.8,
    # ———————————————————————————————————————————    Learning Rate Scheduler     —————————————————————————————————————————————————————
    "lr_scheduler_name": "CosineAnnealingWarmRestarts",
    "gamma": 0.1,
    "step_size": 9,
    "milestones": [1, 3, 5, 7, 8, 9],
    "T_max": 2,
    "T_0": 2,
    "T_mult": 2,
    "mode": "max",
    "patience": 1,
    "factor": 0.5,
    # ————————————————————————————————————————————    Loss And Metric     ———————————————————————————————————————————————————————
    "metric_names": ["DSC"],
    "loss_function_name": "DiceLoss",
 #    "class_weight":  [0.000025, 0.004935, 0.010575, 0.010646, 0.058363, 0.115169, 0.000968, 0.003892,
 # 0.017311, 0.018721, 0.046655, 0.019997, 0.381576, 0.311168],
    "class_weight": [1, 1,1,1,1,1,1,1,1,1,1,1,1,1],
    "sigmoid_normalization": False,
    "dice_loss_mode": "extension",
    "dice_mode": "standard",
    # —————————————————————————————————————————————   Training   ——————————————————————————————————————————————————————
    "optimize_params": False,
    "use_amp": False,
    "run_dir": r"./runs",
    "start_epoch": 0,
    "end_epoch": 100,
    "best_dice": 0,
    "update_weight_freq": 32,
    "terminal_show_freq": 256,
    "save_epoch_freq": 10,
    # ————————————————————————————————————————————   Testing   ———————————————————————————————————————————————————————
    "crop_stride": [32, 32, 32]
}

params_Amos = {
    # ——————————————————————————————————————————————    Launch Initialization    —————————————————————————————————————————————————
    "CUDA_VISIBLE_DEVICES": "0,1,2,3",
    "seed": 1777777,
    "cuda": True,
    "benchmark": False,
    "deterministic": True,
    # —————————————————————————————————————————————     Preprocessing      ————————————————————————————————————————————————————
    "resample_spacing": [ 1.0,
        1.0,
        2.0],
    "clip_lower_bound": -200,
    "clip_upper_bound": 500,
    "samples_train": 3072,
    "samples_valid": 800,
    "crop_size": (96, 96, 96),
    "crop_threshold": 0.1,
    # ——————————————————————————————————————————————    Data Augmentation    ——————————————————————————————————————————————————————
    "augmentation_probability": 0.4,
    "augmentation_method": "Choice",
    "open_elastic_transform": True,
    "elastic_transform_sigma": 20,
    "elastic_transform_alpha": 1,
    "open_gaussian_noise": True,
    "gaussian_noise_mean": 0,
    "gaussian_noise_std": 0.01,
    "open_random_flip": True,
    "open_random_rescale": True,
    "random_rescale_min_percentage": 0.5,
    "random_rescale_max_percentage": 1.5,
    "open_random_rotate": True,
    "random_rotate_min_angle": -50,
    "random_rotate_max_angle": 50,
    "open_random_shift": True,
    "random_shift_max_percentage": 0.3,
    "normalize_mean": 0.456  ,
    "normalize_std": 0.224,
    # —————————————————————————————————————————————    Data Loading     ——————————————————————————————————————————————————————
    "dataset_name": "Amos",
    "dataset_path": r"./datasets/Amos",
    "create_data": False,
    "batch_size": 26,
    "num_workers": 60,
    # —————————————————————————————————————————————    Model     ——————————————————————————————————————————————————————
    "model_name": "SlimFormer",
    "in_channels": 1,
    "classes": 16,
    "scaling_version": "TINY",
    "dimension": "3d",
    "index_to_class_dict": {
        0: "background",
        1: "spleen",
        2: "right kidney",
        3: "left kidney",
        4: "gall bladder",
        5: "esophagus",
        6: "liver",
        7: "stomach",
        8: "arota",
        9: "postcava",
        10: "pancreas",
        11: "right adrenal gland",
        12: "left adrenal gland",
        13: "duodenum",
        14: "bladder",
        15: "prostate/uterus"
    },

    "resume": None,
    "pretrain": None,
    # ——————————————————————————————————————————————    Optimizer     ——————————————————————————————————————————————————————
    "optimizer_name": "Adam",
    "learning_rate": 0.0005,
    "weight_decay": 0.00005,
    "momentum": 0.8,
    # ———————————————————————————————————————————    Learning Rate Scheduler     —————————————————————————————————————————————————————
    "lr_scheduler_name": "CosineAnnealingWarmRestarts",
    "gamma": 0.1,
    "step_size": 9,
    "milestones": [1, 3, 5, 7, 8, 9],
    "T_max": 2,
    "T_0": 2,
    "T_mult": 2,
    "mode": "max",
    "patience": 1,
    "factor": 0.5,
    # ————————————————————————————————————————————    Loss And Metric     ———————————————————————————————————————————————————————
    "metric_names": ["DSC"],
    "loss_function_name": "DiceLoss",
    "class_weight":  [0.000025, 0.004935, 0.010575, 0.010646, 0.058363, 0.115169, 0.000968, 0.003892,
 0.017311, 0.018721, 0.046655, 0.019997, 0.381576, 0.311168],
 #    "class_weight": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    "sigmoid_normalization": False,
    "dice_loss_mode": "extension",
    "dice_mode": "standard",
    # —————————————————————————————————————————————   Training   ——————————————————————————————————————————————————————
    "optimize_params": False,
    "use_amp": False,
    "run_dir": r"./runs",
    "start_epoch": 0,
    "end_epoch": 100,
    "best_dice": 0,
    "update_weight_freq": 32,
    "terminal_show_freq": 256,
    "save_epoch_freq": 4,
    # ————————————————————————————————————————————   Testing   ———————————————————————————————————————————————————————
    "crop_stride": [32, 32, 32]
}

params_Abdomen = {
    # ——————————————————————————————————————————————    Launch Initialization    —————————————————————————————————————————————————
    "CUDA_VISIBLE_DEVICES": "0,1,2,3,5",
    "seed": 1777777,
    "cuda": True,
    "benchmark": False,
    "deterministic": True,
    # —————————————————————————————————————————————     Preprocessing      ————————————————————————————————————————————————————
    "resample_spacing": [ 1.0,
        0.8191796839237213,
        0.8191796839237213],
    "clip_lower_bound": -1412,
    "clip_upper_bound": 17943,
    "samples_train": 4096,
    "samples_valid": 1024,
    "crop_size": (160, 160, 96),
    "crop_threshold": 0.05,
    # ——————————————————————————————————————————————    Data Augmentation    ——————————————————————————————————————————————————————
    "augmentation_probability": 0.3,
    "augmentation_method": "Choice",
    "open_elastic_transform": True,
    "elastic_transform_sigma": 20,
    "elastic_transform_alpha": 1,
    "open_gaussian_noise": True,
    "gaussian_noise_mean": 0,
    "gaussian_noise_std": 0.01,
    "open_random_flip": True,
    "open_random_rescale": True,
    "random_rescale_min_percentage": 0.5,
    "random_rescale_max_percentage": 1.5,
    "open_random_rotate": True,
    "random_rotate_min_angle": -50,
    "random_rotate_max_angle": 50,
    "open_random_shift": True,
    "random_shift_max_percentage": 0.3,
    "normalize_mean": 0.05029342141696459,
    "normalize_std": 0.028477091559295814,
    # —————————————————————————————————————————————    Data Loading     ——————————————————————————————————————————————————————
    "dataset_name": "Abdomen",
    "dataset_path": r"./datasets/Abdomen",
    "create_data": False,
    "batch_size": 8,
    "num_workers": 28,
    # —————————————————————————————————————————————    Model     ——————————————————————————————————————————————————————
    "model_name": "SlimFormer",
    "in_channels": 1,
    "classes": 5,
    "scaling_version": "TINY",
    "dimension": "3d",
    "index_to_class_dict":
    {
        0: "background",
        1: "liver",
        2: "kidney",
        3: "spleen",
        4: "pancreas"
    },
    "resume": None,
    "pretrain": None,
    # ——————————————————————————————————————————————    Optimizer     ——————————————————————————————————————————————————————
    "optimizer_name": "Adam",
    "learning_rate": 0.0005,
    "weight_decay": 0.00005,
    "momentum": 0.8,
    # ———————————————————————————————————————————    Learning Rate Scheduler     —————————————————————————————————————————————————————
    "lr_scheduler_name": "CosineAnnealingWarmRestarts",
    "gamma": 0.1,
    "step_size": 9,
    "milestones": [1, 3, 5, 7, 8, 9],
    "T_max": 2,
    "T_0": 2,
    "T_mult": 2,
    "mode": "max",
    "patience": 1,
    "factor": 0.5,
    # ————————————————————————————————————————————    Loss And Metric     ———————————————————————————————————————————————————————
    "metric_names": ["DSC"],
    "loss_function_name": "DiceLoss",
    "class_weight": [1, 1, 1, 1, 1],
    "sigmoid_normalization": False,
    "dice_loss_mode": "extension",
    "dice_mode": "standard",
    # —————————————————————————————————————————————   Training   ——————————————————————————————————————————————————————
    "optimize_params": False,
    "use_amp": False,
    "run_dir": r"./runs",
    "start_epoch": 0,
    "end_epoch": 20,
    "best_dice": 0,
    "update_weight_freq": 32,
    "terminal_show_freq": 256,
    "save_epoch_freq": 4,
    # ————————————————————————————————————————————   Testing   ———————————————————————————————————————————————————————
    "crop_stride": [32, 32, 32]
}





def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, default="Abdomen", help="dataset name")
    parser.add_argument("-m", "--model", type=str, default="SlimFormer", help="model name")
    parser.add_argument("-pre", "--pretrain_weight", type=str, default=None, help="pre-trained weight file path")
    parser.add_argument("--epoch", type=int, default=None, help="training epoch")
    parser.add_argument("-res", "--resume", type=str, default=None, help="pre-trained state path")

    args = parser.parse_args()

    return args



def main():
    args = parse_args()

    if args.dataset == "Abdomen":
        params = params_Abdomen
    elif args.dataset == "BTCV":
        params = params_BTCV
    elif args.dataset == "Amos":
        params = params_Amos
    else:
        raise RuntimeError(f"No {args.dataset} dataset available")

    params["dataset_name"] = args.dataset
    params["dataset_path"] = os.path.join(r"./datasets", args.dataset)
    params["model_name"] = args.model
    if args.pretrain_weight is not None:
        params["pretrain"] = args.pretrain_weight
    if args.epoch is not None:
        params["end_epoch"] = args.epoch
        params["save_epoch_freq"] = args.epoch // 4


    if params["optimize_params"]:
        tuner_params = nni.get_next_parameter()
        params.update(tuner_params)

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
    utils.reproducibility(params["seed"], params["deterministic"], params["benchmark"])

    # get the cuda device
    if params["cuda"]:
        if torch.cuda.is_available():
            params["device"] = torch.device("cuda")
            device_ids = list(range(torch.cuda.device_count()))
            print(f"Using GPUs: {device_ids}")
        else:
            params["device"] = torch.device("cpu")
            print("No GPUs available, using CPU.")
    else:
        params["device"] = torch.device("cpu")
        print("Using CPU as requested.")
    print(params["device"])
    print("Complete the initialization of configuration")

    # initialize the dataloader
    train_loader, valid_loader = dataloaders.get_dataloader(params)
    print("Complete the initialization of dataloader")


    model, optimizer, lr_scheduler = models.get_model_optimizer_lr_scheduler(params)
    print("Complete the initialization of model:{}, optimizer:{}, and lr_scheduler:{}".format(params["model_name"], params["optimizer_name"], params["lr_scheduler_name"]))

    if device_ids:
        model = model.to(device_ids[0])

        if len(device_ids) > 1:
            model = torch.nn.DataParallel(model, device_ids=device_ids)
    else:
        model = model.to('cpu')

    loss_function = losses.get_loss_function(params)
    print("Complete the initialization of loss function")

    metric = metrics.get_metric(params)
    print("Complete the initialization of metrics")

    trainer = trainers.get_trainer(params, train_loader, valid_loader, model, optimizer, lr_scheduler, loss_function, metric)

    if (params["resume"] is not None) or (params["pretrain"] is not None):
        trainer.load()
    print("Complete the initialization of trainer")

    trainer.training()



if __name__ == '__main__':
    main()


