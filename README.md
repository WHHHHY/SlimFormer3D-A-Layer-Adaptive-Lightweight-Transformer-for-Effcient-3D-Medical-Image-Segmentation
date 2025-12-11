# SlimFormer3D-A-Layer-Adaptive-Lightweight-Transformer-for-Effcient-3D-Medical-Image-Segmentation
## 📘 Paper
**Title:** *SlimFormer3D-A-Layer-Adaptive-Lightweight-Transformer-for-Effcient-3D-Medical-Image-Segmentation*  
**Link:** [(https://papers.miccai.org/miccai-2025/paper/1666_paper.pdf)](https://papers.miccai.org/miccai-2025/paper/1666_paper.pdf)

<p align="center">
  <img src="framework.png" alt="Example Image" width="800">
</p>

## 🏋️ Training
To train the model on a selected dataset, run the following command:

```bash
python ./train.py --dataset Abdomen --model SlimFormer --epoch 150
```

## 📚 Dataset
This project uses the following publicly available abdominal CT segmentation datasets:
This project utilizes the following publicly available abdominal CT segmentation datasets:

- **AbdomenCT-1K**: A large-scale dataset, annotated with multiple abdominal organs for segmentation tasks.
- **BTCV (Beyond the Cranial Vault Challenge)**: A collection of abdominal CT scans used in the **BTCV 2015 challenge**, designed for multi-organ segmentation and evaluation.
- **AMOS (Abdominal Multi-Organ Segmentation)**: A dataset containing **CT images** with annotations for multiple abdominal organs, widely used for evaluating abdominal segmentation algorithms.

## 📑 Citation
If you find this work helpful, please consider citing our paper:

```bibtex
@inproceedings{hong2025slimformer,
  title={SlimFormer-3D: A Layer-Adaptive Lightweight Transformer for Efficient 3D Medical Image Segmentation},
  author={Hong, Yang and Zhang, Lei and Ye, Xujiong and Mo, Jianqing},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={537--546},
  year={2025},
  organization={Springer}
}
