# Pytorch Headtrip

-   Work in Progress

![](examples/dream_example.gif)

[![](https://img.youtube.com/vi/Cd5LNeT5wHI/0.jpg)](https://youtu.be/Cd5LNeT5wHI)

Single Deep Dreaming and Sequence Dreaming with Optical Flow and Depth Estimation in Pytorch.

Features:

-   Sequence Dreaming with Optical Flow (Farneback or SpyNet)
-   Depth Estimation with [MiDas](https://pytorch.org/hub/intelisl_midas_v2/)
-   Supports multiple Pytorch Architectures
-   Dream Single Class of ImageNet, check my post [here](https://towardsdatascience.com/deep-lucid-dreaming-94fecd3cd46d)

# Install

## Requirements

1. Python 3.7
2. Pytorch 1.7
3. OpenCV
4. Matplotlib

```
pip install -r requirements.txt
```
Depending on your cuda version, you might get an error installing pytorch in your env.
If you want to use fp16, you need to install _[apex](https://github.com/NVIDIA/apex)_ manually

# Usage

For inference you need a config file like the basic*conf in the \_configs* folder.

```
python dream.py --config configs/basic_conf.yaml
```

Currently, its only possible to dream on a sequence of frames from a video that are
extracted beforehand with e.g. _ffmpeg_

-   The SpyNet Code is adapted from this [github repository](https://github.com/sniklaus/pytorch-spynet)
