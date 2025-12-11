from .abdomen_trainer import AbdomenTrainer
from .btcv_trainer import BTCVTrainer
from .amos_trainer import AmosTrainer
def get_trainer(opt, train_loader, valid_loader, model, optimizer, lr_scheduler, loss_function, metric):
    if opt["dataset_name"] == "Abdomen":
        trainer = AbdomenTrainer(opt, train_loader, valid_loader, model, optimizer, lr_scheduler, loss_function, metric)
    elif opt["dataset_name"] == "BTCV":
        trainer = BTCVTrainer(opt, train_loader, valid_loader, model, optimizer, lr_scheduler, loss_function, metric)
    elif opt["dataset_name"] == "Amos":
        trainer = AmosTrainer(opt, train_loader, valid_loader, model, optimizer, lr_scheduler, loss_function, metric)
    else:
        raise RuntimeError(f"No {opt['dataset_name']} dataset available when initialize trainer")

    return trainer
