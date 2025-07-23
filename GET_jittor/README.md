# GET: Unlocking the Multi-modal Potential of CLIP for Generalized Category Discovery (CVPR2025)

<p align="center">
    <a href="https://cvpr.thecvf.com/virtual/2025/poster/34519"><img src="https://img.shields.io/badge/CVPR%202025-68488b"></a>
    <a href="https://arxiv.org/abs/2403.09974"><img src="https://img.shields.io/badge/arXiv-2403.09974-b31b1b"></a>
  <a href="https://github.com/enguangW/GET/blob/master/LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue.svg"></a>
</p>
<p align="center">
	 This is the jittor implementation for our paper our CVPR2025 paper "GET: Unlocking the Multi-modal Potential of CLIP for Generalized Category Discovery" <br>
</p>



## 🚀 Getting Started

### 1. Datasets

We use the standard benchmarks in this paper, including:

* Generic datasets：[CIFAR-10/100](https://pytorch.org/vision/stable/datasets.html) and [ImageNet-100/1K](https://image-net.org/download.php)

* Fine-grained datasets: [The Semantic Shift Benchmark (SSB)](https://github.com/sgvaze/osr_closed_set_all_you_need#ssb) and [Herbarium19](https://www.kaggle.com/c/herbarium-2019-fgvc6)

### 2. Dependencies

Refer to [GET](https://github.com/enguangW/GET) and [JCLIP](https://github.com/uyzhang/JCLIP)




### 4. Train the model
First, convert PyTorch ckpt into Jittor ckpt:

```
python change_ckpt_jittor.py

```


Train TES on the CUB dataset: 

```
python s1_TES.py

```
Train GET:
```
python s2_GET.py
```


