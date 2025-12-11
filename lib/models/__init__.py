# -*- encoding: utf-8 -*-
"""
@author   :   yykzjh    
@Contact  :   yykzhjh@163.com
@DateTime :   2023/12/30 16:57
@Version  :   1.0
@License  :   (C)Copyright 2023
"""
import torch.optim as optim
import lib.utils as utils
from .SlimFormer import SlimFormer

def get_model_optimizer_lr_scheduler(opt):
    # initialize model
    if opt["dataset_name"] == "Abdomen" :
        if opt["model_name"] == "SlimFormer":
            model = SlimFormer(in_channels=opt["in_channels"], out_channels=opt["classes"])
        else:
            raise RuntimeError(f"No {opt['model_name']} model available on {opt['dataset_name']} dataset")
    elif opt["dataset_name"] == "BTCV":
        if opt["model_name"] == "SlimFormer":
            model = SlimFormer(in_channels=opt["in_channels"], out_channels=opt["classes"])
        else:
            raise RuntimeError(f"No {opt['model_name']} model available on {opt['dataset_name']} dataset")
    elif opt["dataset_name"] == "Amos":
        if opt["model_name"] == "SlimFormer":
            model = SlimFormer(in_channels=opt["in_channels"], out_channels=opt["classes"])
        else:
            raise RuntimeError(f"No {opt['model_name']} model available on {opt['dataset_name']} dataset")
    else:
        raise RuntimeError(f"No {opt['dataset_name']} dataset available when initialize model")

    # initialize model and weights
    model = model.to(opt["device"])
    utils.init_weights(model, init_type="kaiming")

    # initialize optimizer
    if opt["optimizer_name"] == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=opt["learning_rate"], momentum=opt["momentum"],
                              weight_decay=opt["weight_decay"])

    elif opt["optimizer_name"] == 'Adagrad':
        optimizer = optim.Adagrad(model.parameters(), lr=opt["learning_rate"], weight_decay=opt["weight_decay"])

    elif opt["optimizer_name"] == "RMSprop":
        optimizer = optim.RMSprop(model.parameters(), lr=opt["learning_rate"], weight_decay=opt["weight_decay"],
                                  momentum=opt["momentum"])

    elif opt["optimizer_name"] == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=opt["learning_rate"], weight_decay=opt["weight_decay"])

    elif opt["optimizer_name"] == "AdamW":
        optimizer = optim.AdamW(model.parameters(), lr=opt["learning_rate"], weight_decay=opt["weight_decay"])

    elif opt["optimizer_name"] == "Adamax":
        optimizer = optim.Adamax(model.parameters(), lr=opt["learning_rate"], weight_decay=opt["weight_decay"])

    elif opt["optimizer_name"] == "Adadelta":
        optimizer = optim.Adadelta(model.parameters(), lr=opt["learning_rate"], weight_decay=opt["weight_decay"])

    else:
        raise RuntimeError(f"No {opt['optimizer_name']} optimizer available")

    # initialize lr_scheduler
    if opt["lr_scheduler_name"] == "ExponentialLR":
        lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=opt["gamma"])

    elif opt["lr_scheduler_name"] == "StepLR":
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=opt["step_size"], gamma=opt["gamma"])

    elif opt["lr_scheduler_name"] == "MultiStepLR":
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=opt["milestones"], gamma=opt["gamma"])

    elif opt["lr_scheduler_name"] == "CosineAnnealingLR":
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt["T_max"])

    elif opt["lr_scheduler_name"] == "CosineAnnealingWarmRestarts":
        lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=opt["T_0"],
                                                                      T_mult=opt["T_mult"])

    elif opt["lr_scheduler_name"] == "OneCycleLR":
        lr_scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=opt["learning_rate"],
                                                     steps_per_epoch=opt["steps_per_epoch"], epochs=opt["end_epoch"], cycle_momentum=False)

    elif opt["lr_scheduler_name"] == "ReduceLROnPlateau":
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=opt["mode"], factor=opt["factor"],
                                                            patience=opt["patience"])
    else:
        raise RuntimeError(f"No {opt['lr_scheduler_name']} lr_scheduler available")

    return model, optimizer, lr_scheduler


def get_model(opt):
    # initialize model
    if opt["dataset_name"] == "Abdomen":
        if opt["model_name"] == "SlimFormer":
            model = SlimFormer(in_channels=opt["in_channels"], out_channels=opt["classes"])
        else:
            raise RuntimeError(f"No {opt['model_name']} model available on {opt['dataset_name']} dataset")
    elif opt["dataset_name"] == "BTCV":
        if opt["model_name"] == "SlimFormer":
            model = SlimFormer(in_channels=opt["in_channels"], out_channels=opt["classes"])
        else:
            raise RuntimeError(f"No {opt['model_name']} model available on {opt['dataset_name']} dataset")

    else:
        raise RuntimeError(f"No {opt['dataset_name']} dataset available when initialize model")

    model = model.to(opt["device"])

    return model
