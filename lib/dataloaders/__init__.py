from torch.utils.data import DataLoader

from .AbdomenDataset import AbdomenDataset
from .BTCVDataset import BTCVDataset
from .AmosDataset import AmosDataset

def get_dataloader(opt):
    if opt["dataset_name"] == "Abdomen":
        train_set = AbdomenDataset(opt, mode="train")
        valid_set = AbdomenDataset(opt, mode="valid")

        train_loader = DataLoader(train_set, batch_size=opt["batch_size"], shuffle=True, num_workers=opt["num_workers"], pin_memory=True)
        valid_loader = DataLoader(valid_set, batch_size=1, shuffle=False, num_workers=28, pin_memory=True)
    elif opt["dataset_name"] == "BTCV":
        train_set = BTCVDatasetDataset(opt, mode="train")
        valid_set = BTCVDataset(opt, mode="valid")

        train_loader = DataLoader(train_set, batch_size=opt["batch_size"], shuffle=True, num_workers=opt["num_workers"], pin_memory=True)
        valid_loader = DataLoader(valid_set, batch_size=1, shuffle=False, num_workers=30, pin_memory=True)
    elif opt["dataset_name"] == "Amos":
        train_set = AmosDataset(opt, mode="train")
        valid_set = AmosDataset(opt, mode="valid")

        train_loader = DataLoader(train_set, batch_size=opt["batch_size"], shuffle=True, num_workers=opt["num_workers"], pin_memory=True)
        valid_loader = DataLoader(valid_set, batch_size=1, shuffle=False, num_workers=60, pin_memory=True)
    else:
        raise RuntimeError(f"No {opt['dataset_name']} dataloader available")

    opt["steps_per_epoch"] = len(train_loader)

    return train_loader, valid_loader


def get_test_dataloader(opt):
    if opt["dataset_name"] == "Abdomen":
        valid_set = AbdomenDataset(opt, mode="valid")
        valid_loader = DataLoader(valid_set, batch_size=opt["batch_size"], shuffle=False, num_workers=1, pin_memory=True)
    elif opt["dataset_name"] == "Amos":
        valid_set = AmosDataset(opt, mode="valid")
        valid_loader = DataLoader(valid_set, batch_size=opt["batch_size"], shuffle=False, num_workers=1, pin_memory=True)
    elif opt["dataset_name"] == "BTCV":
        valid_set = BTCVDataset(opt, mode="valid")
        valid_loader = DataLoader(valid_set, batch_size=opt["batch_size"], shuffle=False, num_workers=1, pin_memory=True)
    else:
        raise RuntimeError(f"No {opt['dataset_name']} dataloader available")

    return valid_loader
