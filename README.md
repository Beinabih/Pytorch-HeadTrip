# Pytorch Headtrip

![](examples/dream_example.gif)

- Work in Progress	

Single Deep Dreaming and Sequence Dreaming with Optical Flow and Depth Estimation in Pytorch. 


Features:
- Sequence Dreaming with Optical Flow
- Depth Estimation with MiDas
- Supports multiple Pytorch Architectures
- Dream Single Class of ImageNet


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

