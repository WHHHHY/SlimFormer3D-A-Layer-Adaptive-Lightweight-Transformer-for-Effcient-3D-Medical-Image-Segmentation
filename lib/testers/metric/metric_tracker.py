from __future__ import annotations
from typing import Dict, Any, Iterable
import numpy as np
import torch


class MetricTracker:

    def __init__(
        self,
        classes: int,
        index_to_class_dict: Dict[int, str],
        metric_names: Iterable[str],
        metrics_impl: Dict[str, Any],
    ) -> None:
        self.classes = int(classes)
        self.index_to_class_dict = dict(index_to_class_dict)
        self.metric_names = list(metric_names)
        self.metrics_impl = dict(metrics_impl) if metrics_impl is not None else {}
        self.statistics_dict = self._init_statistics_dict()

    def _init_statistics_dict(self) -> Dict[str, Any]:
        statistics_dict: Dict[str, Any] = {
            metric_name: {class_name: 0.0 for _, class_name in self.index_to_class_dict.items()}
            for metric_name in self.metric_names
        }
        for metric_name in self.metric_names:
            statistics_dict[metric_name]["avg"] = 0.0
        statistics_dict["total_area_intersect"] = np.zeros((self.classes,), dtype=np.float64)
        statistics_dict["total_area_union"] = np.zeros((self.classes,), dtype=np.float64)
        statistics_dict["class_count"] = {
            class_name: 0 for _, class_name in self.index_to_class_dict.items()
        }
        statistics_dict["count"] = 0
        return statistics_dict

    def reset(self) -> None:
        self.statistics_dict["count"] = 0
        self.statistics_dict["total_area_intersect"] = np.zeros((self.classes,), dtype=np.float64)
        self.statistics_dict["total_area_union"] = np.zeros((self.classes,), dtype=np.float64)
        for _, class_name in self.index_to_class_dict.items():
            self.statistics_dict["class_count"][class_name] = 0
        for metric_name in self.metric_names:
            self.statistics_dict[metric_name]["avg"] = 0.0
            for _, class_name in self.index_to_class_dict.items():
                self.statistics_dict[metric_name][class_name] = 0.0

    def update_batch(self, output: torch.Tensor, target: torch.Tensor) -> None:
        if output.ndim < 4:
            raise ValueError(f"Unexpected output shape: {tuple(output.shape)}")
        if target.ndim != output.ndim - 1:
            raise ValueError(f"Unexpected target shape: {tuple(target.shape)}")
        B, K = output.shape[0], output.shape[1]
        if target.shape[0] != B:
            raise ValueError("Batch size mismatch")
        if target.shape[1:] != output.shape[2:]:
            raise ValueError("Spatial shape mismatch")

        cur_batch_size = int(target.shape[0])
        device = target.device
        mask = torch.zeros(self.classes, device=device)
        unique_index = torch.unique(target).int()
        for index in unique_index:
            if 0 <= int(index) < self.classes:
                mask[index] = 1

        self.statistics_dict["count"] += cur_batch_size

        for i, class_name in self.index_to_class_dict.items():
            if 0 <= int(i) < self.classes and mask[i] == 1:
                self.statistics_dict["class_count"][class_name] += cur_batch_size

        for metric_name in self.metric_names:
            if metric_name not in self.metrics_impl:
                continue
            metric_fn = self.metrics_impl[metric_name]

            if metric_name == "IoU":
                area_intersect, area_union, _, _ = metric_fn(output, target)
                area_intersect = area_intersect.detach().cpu().numpy()
                area_union = area_union.detach().cpu().numpy()
                self.statistics_dict["total_area_intersect"] += area_intersect
                self.statistics_dict["total_area_union"] += area_union
            else:
                per_class_metric = metric_fn(output, target)
                per_class_metric = per_class_metric.to(device)
                effective_mask = mask.to(per_class_metric.device)
                if torch.sum(effective_mask) > 0:
                    per_class_metric = per_class_metric * effective_mask
                    batch_avg = (torch.sum(per_class_metric) / torch.sum(effective_mask)).item()
                    self.statistics_dict[metric_name]["avg"] += batch_avg * cur_batch_size
                for j, class_name in self.index_to_class_dict.items():
                    if 0 <= int(j) < per_class_metric.numel():
                        self.statistics_dict[metric_name][class_name] += (
                            per_class_metric[j].item() * cur_batch_size
                        )

    def get_class_IoU(self) -> np.ndarray:
        inter = self.statistics_dict["total_area_intersect"]
        union = self.statistics_dict["total_area_union"]
        class_IoU = inter / (union + 1e-12)
        class_IoU = np.nan_to_num(class_IoU)
        return class_IoU

    def get_mIoU(self) -> float:
        class_IoU = self.get_class_IoU()
        return float(np.mean(class_IoU))

    def summary(self) -> Dict[str, Any]:
        class_IoU = self.get_class_IoU()
        mIoU = self.get_mIoU()
        metrics_out: Dict[str, Any] = {}
        for metric_name in self.metric_names:
            if metric_name == "IoU":
                continue
            avg_val = 0.0
            if self.statistics_dict["count"] != 0:
                avg_val = (
                    self.statistics_dict[metric_name]["avg"] / self.statistics_dict["count"]
                )
            per_class_vals = {}
            for _, class_name in self.index_to_class_dict.items():
                cnt = self.statistics_dict["class_count"][class_name]
                if cnt != 0:
                    per_class_vals[class_name] = (
                        self.statistics_dict[metric_name][class_name] / cnt
                    )
                else:
                    per_class_vals[class_name] = float("nan")
            metrics_out[metric_name] = {
                "avg": float(avg_val),
                "per_class": per_class_vals,
            }
        return {
            "class_IoU": class_IoU,
            "mIoU": mIoU,
            "metrics": metrics_out,
        }

    def print_summary(self) -> None:
        print("\n" + "=" * 80)
        print("EVALUATION RESULTS")
        print("=" * 80)
        class_IoU = self.get_class_IoU()
        mIoU = self.get_mIoU()
        print_info = ""
        print_info += " " * 12
        for metric_name in self.metric_names:
            print_info += "{:^12}".format(metric_name)
        print_info += "{:^12}".format("IoU")
        print_info += "\n"
        for i, class_name in self.index_to_class_dict.items():
            print_info += "{:<12}".format(class_name)
            for metric_name in self.metric_names:
                if metric_name != "IoU":
                    value = 0.0
                    if self.statistics_dict["class_count"][class_name] != 0:
                        value = (
                            self.statistics_dict[metric_name][class_name]
                            / self.statistics_dict["class_count"][class_name]
                        )
                    print_info += "{:^12.6f}".format(value)
            print_info += "{:^12.6f}".format(class_IoU[i])
            print_info += "\n"
        print_info += "{:<12}".format("average")
        for metric_name in self.metric_names:
            if metric_name != "IoU":
                value = 0.0
                if self.statistics_dict["count"] != 0:
                    value = (
                        self.statistics_dict[metric_name]["avg"]
                        / self.statistics_dict["count"]
                    )
                print_info += "{:^12.6f}".format(value)
        print_info += "{:^12.6f}".format(mIoU)
        print(print_info)
        print("=" * 80)
        for metric_name in self.metric_names:
            if metric_name == "IoU":
                continue
            msg = [f"{metric_name}:"]
            for _, class_name in self.index_to_class_dict.items():
                cnt = self.statistics_dict["class_count"][class_name]
                if cnt:
                    metric_val = self.statistics_dict[metric_name][class_name] / cnt
                else:
                    metric_val = float("nan")
                msg.append(f"{class_name}={metric_val:.4f}")
            print("  ".join(msg))
