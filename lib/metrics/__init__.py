from lib.metrics import Abdomen
from lib.metrics import BTCV
from lib.metrics import Amos


def get_metric(opt):
    if opt["dataset_name"] == "Abdomen":
        metrics = []
        for metric_name in opt["metric_names"]:
            if metric_name == "DSC":
                metrics.append(Tooth.DICE(num_classes=opt["classes"], sigmoid_normalization=opt["sigmoid_normalization"], mode=opt["dice_mode"]))

            elif metric_name == "ASSD":
                metrics.append(Tooth.AverageSymmetricSurfaceDistance(num_classes=opt["classes"], sigmoid_normalization=opt["sigmoid_normalization"]))

            elif metric_name == "HD":
                metrics.append(Tooth.HausdorffDistance(num_classes=opt["classes"], sigmoid_normalization=opt["sigmoid_normalization"]))

            elif metric_name == "SO":
                metrics.append(Tooth.SurfaceOverlappingValues(num_classes=opt["classes"], sigmoid_normalization=opt["sigmoid_normalization"], theta=1.0))

            elif metric_name == "SD":
                metrics.append(Tooth.SurfaceDice(num_classes=opt["classes"], sigmoid_normalization=opt["sigmoid_normalization"], theta=1.0))

            elif metric_name == "IoU":
                metrics.append(Tooth.IoU(num_classes=opt["classes"], sigmoid_normalization=opt["sigmoid_normalization"]))

            else:
                raise RuntimeError(f"No {metric_name} metric available on {opt['dataset_name']} dataset")
    elif opt["dataset_name"] == "BTCV":
        metrics = []
        for metric_name in opt["metric_names"]:
            if metric_name == "DSC":
                metrics.append(
                    Abdomen.DICE(num_classes=opt["classes"], sigmoid_normalization=opt["sigmoid_normalization"],
                                 mode=opt["dice_mode"]))

            elif metric_name == "ASSD":
                metrics.append(Tooth.AverageSymmetricSurfaceDistance(num_classes=opt["classes"],
                                                                     sigmoid_normalization=opt[
                                                                         "sigmoid_normalization"]))

            elif metric_name == "HD":
                metrics.append(Tooth.HausdorffDistance(num_classes=opt["classes"],
                                                       sigmoid_normalization=opt["sigmoid_normalization"]))

            elif metric_name == "SO":
                metrics.append(Tooth.SurfaceOverlappingValues(num_classes=opt["classes"],
                                                              sigmoid_normalization=opt["sigmoid_normalization"],
                                                              theta=1.0))

            elif metric_name == "SD":
                metrics.append(
                    Tooth.SurfaceDice(num_classes=opt["classes"], sigmoid_normalization=opt["sigmoid_normalization"],
                                      theta=1.0))

            elif metric_name == "IoU":
                metrics.append(
                    Tooth.IoU(num_classes=opt["classes"], sigmoid_normalization=opt["sigmoid_normalization"]))

            else:
                raise RuntimeError(f"No {metric_name} metric available on {opt['dataset_name']} dataset")
    elif opt["dataset_name"] == "Amos":
        metrics = []
        for metric_name in opt["metric_names"]:
            if metric_name == "DSC":
                metrics.append(
                    Abdomen.DICE(num_classes=opt["classes"], sigmoid_normalization=opt["sigmoid_normalization"],
                                 mode=opt["dice_mode"]))

            elif metric_name == "ASSD":
                metrics.append(Tooth.AverageSymmetricSurfaceDistance(num_classes=opt["classes"],
                                                                     sigmoid_normalization=opt[
                                                                         "sigmoid_normalization"]))

            elif metric_name == "HD":
                metrics.append(Tooth.HausdorffDistance(num_classes=opt["classes"],
                                                       sigmoid_normalization=opt["sigmoid_normalization"]))

            elif metric_name == "SO":
                metrics.append(Tooth.SurfaceOverlappingValues(num_classes=opt["classes"],
                                                              sigmoid_normalization=opt["sigmoid_normalization"],
                                                              theta=1.0))

            elif metric_name == "SD":
                metrics.append(
                    Tooth.SurfaceDice(num_classes=opt["classes"], sigmoid_normalization=opt["sigmoid_normalization"],
                                      theta=1.0))

            elif metric_name == "IoU":
                metrics.append(
                    Tooth.IoU(num_classes=opt["classes"], sigmoid_normalization=opt["sigmoid_normalization"]))

            else:
                raise RuntimeError(f"No {metric_name} metric available on {opt['dataset_name']} dataset")
    else:
        raise RuntimeError(f"No {opt['dataset_name']} dataset available when initialize metrics")

    return metrics