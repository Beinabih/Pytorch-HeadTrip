# Pytorch Headtrip

![](examples/dream_example.gif)

- Work in Progress	

Single Deep Dreaming and Sequence Dreaming with Optical Flow and Depth Estimation in Pytorch. 


Features:
- Sequence Dreaming with Optical Flow (Farneback or SpyNet)
- Depth Estimation with [MiDas](https://pytorch.org/hub/intelisl_midas_v2/)
- Supports multiple Pytorch Architectures
- Dream Single Class of ImageNet, check my post [here](https://towardsdatascience.com/deep-lucid-dreaming-94fecd3cd46d)


# Install

```
pip install -r requirements.txt
```

If you want to use fp16, you need to install apex manually

# Usage

For inference you need a config file like the basic_conf in the *configs* folder. 

```
python dream.py --config configs/basic_conf.yaml
```

Currently, its only possible to dream on each extracted frame from a video that is
extracted beforehand with e.g. *ffmpeg*

