# Pytorch Headtrip

-   Work in Progress

![](examples/dream_example.gif)

[![](https://img.youtube.com/vi/Cd5LNeT5wHI/0.jpg)](https://youtu.be/Cd5LNeT5wHI)

Single Deep Dreaming and Sequence Dreaming with Optical Flow and Depth Estimation in Pytorch.

Check out my [article](https://towardsdatascience.com/sequence-dreaming-with-depth-estimation-in-pytorch-d754cba14d30) for the setup!

[@aertist](https://github.com/aertist) made a colab notebook: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1hZFeBtLTY1nUBxC0G_qvbz4ZilUh67rr?usp=sharing)

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

# Usage

For inference you need a config file like the basic*conf in the \_configs* folder.

```
python dream.py --config configs/basic_conf.yaml
```

Currently, its only possible to dream on a sequence of frames from a video that are
extracted beforehand with e.g. _ffmpeg_

-   The SpyNet Code is adapted from this [github repository](https://github.com/sniklaus/pytorch-spynet)

[**Buy me a coffee! :coffee:**](https://www.buymeacoffee.com/beinabih)
