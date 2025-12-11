import os
import argparse
import torch
from lib import utils, dataloaders, models, metrics, testers
params_Abdomen = {
    # ——————————————————————————————————————————————    Launch Initialization    —————————————————————————————————————————————————
    "CUDA_VISIBLE_DEVICES": "0,1",
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
    "samples_train": 2048,
    "crop_size": (512, 512, 110),
    "crop_threshold": 0.5,
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
    "dataset_name": "AbdomenCT",
    "dataset_path": r"./datasets/AbdomenCT",
    "create_data": False,
    "batch_size": 32,
    "num_workers": 16,
    # —————————————————————————————————————————————    Model     ——————————————————————————————————————————————————————
    "model_name": "SlimFormer",
    "in_channels": 1,
    "classes": 2,
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
    "lr_scheduler_name": "ReduceLROnPlateau",
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
    "class_weight": [0.1, 0.3, 0.3, 0.2, 0.1],
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
params_BTCV = {
    # ——————————————————————————————————————————————    Launch Initialization    —————————————————————————————————————————————————
    "CUDA_VISIBLE_DEVICES": "0,1,2,3",
    "seed": 1777777,
    "cuda": True,
    "benchmark": False,
    "deterministic": True,
    # —————————————————————————————————————————————     Preprocessing      ————————————————————————————————————————————————————
    "resample_spacing": [1.5, 1.5, 2.0],
    "clip_lower_bound": -200,
    "clip_upper_bound": 500,
    "samples_train": 3072,
    "samples_valid": 340,
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
    "random_rotate_min_angle": -30,
    "random_rotate_max_angle": 30,
    "open_random_shift": True,
    "random_shift_max_percentage": 0.2,
    "normalize_mean": 0.456,
    "normalize_std": 0.224,
    # —————————————————————————————————————————————    Data Loading     ——————————————————————————————————————————————————————
    "dataset_name": "BTCV",
    "dataset_path": r"./datasets/BTCV",
    "create_data": False,
    "batch_size": 1,
    "num_workers": 60,
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
    "class_weight": [1, 1,1,1,1,1,1,1,1,1,1,1,1,1],
    "sigmoid_normalization": False,
    "dice_loss_mode": "extension",
    "dice_mode": "standard",
    # —————————————————————————————————————————————   Training   ——————————————————————————————————————————————————————
    "optimize_params": False,
    "use_amp": False,
    "run_dir": r"./runs",
    "start_epoch": 0,
    "end_epoch": 400,
    "best_dice": 0,
    "update_weight_freq": 4,
    "terminal_show_freq": 256,
    "save_epoch_freq": 10,
    # ————————————————————————————————————————————   Testing   ———————————————————————————————————————————————————————
    "crop_stride": [32, 32, 32]
}



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="Abdomen", help="dataset name")
    parser.add_argument("--model", type=str, default="SlimFormer", help="model name")
    parser.add_argument("--pretrain_weight", type=str, default=None, help="pre-trained weight file path")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    if args.dataset == "Abdomen":
        params = params_Abdomen
    elif args.dataset == "BTCV":
        params = params_BTCV
    else:
        raise RuntimeError(f"No {args.dataset} dataset available")

    params["dataset_name"] = args.dataset
    params["dataset_path"] = os.path.join(r"./datasets", args.dataset)
    params["model_name"] = args.model
    if args.pretrain_weight is None:
        raise RuntimeError("model weights cannot be None")
    params["pretrain"] = args.pretrain_weight
    params["dimension"] = args.dimension
    params["scaling_version"] = args.scaling_version

    os.environ["CUDA_VISIBLE_DEVICES"] = params["CUDA_VISIBLE_DEVICES"]
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
    utils.reproducibility(params["seed"], params["deterministic"], params["benchmark"])

    if params["cuda"]:
        params["device"] = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        params["device"] = torch.device("cpu")
    print(params["device"])
    print("Complete the initialization of configuration")

    # initialize the dataloader
    valid_loader = dataloaders.get_test_dataloader(params)
    print("Complete the initialization of dataloader")

    model = models.get_model(params)
    print("Complete the initialization of model:{}".format(params["model_name"]))

    metric = metrics.get_metric(params)
    print("Complete the initialization of metrics")

    tester = testers.get_tester(params, model, metric)
    print("Complete the initialization of tester")

    tester.load()
    print("Complete loading training weights")

    tester.evaluation(valid_loader)


if __name__ == '__main__':
    main()
