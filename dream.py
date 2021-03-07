"""Summary
"""
import argparse
import glob
import os
import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as nd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import tqdm
from PIL import Image
from torch.autograd import Variable
from torchvision import models

from config import load_config
from depth_inference import MiDaS
from spyNet import calc_opflow
from utils import clip, convert, deprocess, get_octaves, preprocess, warp


class Dreamer:
    def __init__(self, img_p, outpath, config):
        """Summary

        Args:
            model (TYPE): Description
            batchsize (TYPE): Description
            img_p (TYPE): Description
            outpath (TYPE): Description
            config (TYPE): Description
        """
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.config = config
        self.img_p = img_p
        self.outpath = outpath
        self.init_model()

        # self.layers = list(model.children())
        self.octave_list = [1.1, 1.2, 1.3, 1.4, 1.5]
        self.num_octaves_para = config["num_octaves"]
        self.octave_scale = config["octave_scale"]
        self.at_layer_para = config["at_layer"]
        self.lr = config["lr"]
        self.random = config["random"]
        self.no_class = config["no_class"]
        self.ch_list = config["channel_list"]
        self.img_list = sorted(glob.glob(img_p))
        self.depth = config["use_depth"]
        self.depth_w = config["depth_str"]

        self.depth_model = MiDaS(False)

        self.loss = nn.BCEWithLogitsLoss()

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

        if self.random:
            self.random_para()

        if self.no_class:
            self.model = nn.Sequential(*self.layers[: (self.at_layer_para + 1)])

        print(
            self.config["num_iterations"],
            self.octave_scale,
            self.config["num_octaves"],
            self.lr,
        )

    def init_model(self):
        """initializes the model with the config file"""
        if self.config["model"] == "resnet":
            network = models.resnext50_32x4d(pretrained=True)
        elif self.config["model"] == "vgg19":
            network = models.vgg19(pretrained=True)
            print(network)
        elif self.config["model"] == "densenet":
            network = models.densenet121(pretrained=True)
        elif self.config["model"] == "inception":
            network = models.inception_v3(pretrained=True)
        elif self.config["model"] == "mobile":
            network = models.mobilenet_v2(pretrained=True)
        elif self.config["model"] == "shuffle":
            network = models.shufflenet_v2_x0_5(pretrained=True)
        elif self.config["model"] == "squeeze":
            network = models.squeezenet1_1(pretrained=True)
        elif self.config["model"] == "resnetx":
            network = models.resnext101_32x8d(pretrained=True)
        elif self.config["model"] == "masnet":
            network = models.mnasnet1_0(pretrained=True)
        elif self.config["model"] == "googlenet":
            network = models.googlenet(pretrained=True)
        elif self.config["model"] == "alexnet":
            network = models.alexnet(pretrained=True)
        else:
            print("Invalid Model")

        network.eval()
        self.model = network.to(self.device)

        if self.config["fp16"]:
            amp.register_float_function(torch, "batch_norm")
            self.model = amp.initialize(self.model, opt_level="O2")

    def forward(self, model, image, z, d_img=None, mask=None):
        """Summary

        Args:
            model (TYPE): Description
            image (TYPE): Description
            z (TYPE): Description
            d_img (None, optional): Description
            mask (None, optional): Description

        Returns:
            TYPE: Description
        """
        model.zero_grad()
        out = model(image)

        if self.config["guided"]:
            target = self.get_target(self.config, z, out)
            loss = -self.loss(out, target)
        else:
            loss = out.norm()

        loss.backward()

        avg_grad = np.abs(image.grad.data.cpu().numpy()).mean()
        norm_lr = self.lr / avg_grad
        grad = image.grad.data

        dream_grad = grad * (norm_lr * 5)

        if self.depth:
            d_img = torch.from_numpy(d_img)
            d_img = d_img[0, 0].to(self.device)

            dream_grad *= d_img * self.depth_w

        if mask is not None:
            mask = torch.from_numpy(mask)
            dream_grad *= mask.to(self.device)

        image.data += dream_grad
        image.data = clip(image.data)
        image.grad.data.zero_()

        return image

    def get_target(self, config, z, out):
        """Summary

        Args:
            config (Dictionary): Config File
            z (Integer): Iteration Value
            out (tensor): Model Output

        Returns:
            Tensor: Target Tensor for guided dreaming
        """
        target = torch.zeros((1, 1000)).to(self.device)

        if config["max_output"]:
            out = out.float()

            if config["pyramid_max"]:
                if z == 0:
                    self.channel = out.argmax()
                    target[0, self.channel] = 100
                else:
                    out[0, self.channel] = 0
                    self.channel = out.argmax()
                    target = torch.zeros((1, 1000)).to(self.device)
                    target[0, self.channel] = 100
            else:
                self.channel = out.argmax()
                target[0, self.channel] = 100

        else:
            for ch in config["channel_list"]:
                target[0, ch] = 100

        return target

    def dream(self, image, model, d_img=None, mask=None):
        """Updates the image to maximize outputs for n iterations

        Args:
            image (TYPE): Description
            model (TYPE): Description
            d_img (None, optional): Description
            mask (None, optional): Description

        Returns:
            TYPE: Description
        """
        Tensor = (
            torch.cuda.FloatTensor if torch.cuda.is_available else torch.FloatTensor
        )
        image = Variable(Tensor(image), requires_grad=True)

        for n in range(self.config["num_iterations"]):
            image = self.forward(model, image, n, d_img, mask)

        return image.cpu().data.numpy()

    def deep_dream(self, image, model, i, seq, mask=None):
        """Main deep dream method

        Args:
            image (TYPE): Description
            model (TYPE): Description
            i (TYPE): Description
            seq (TYPE): Description
            mask (None, optional): Description

        Returns:
            TYPE: Description
        """

        image_p = image.unsqueeze(0).cpu().detach().numpy()

        octaves = get_octaves(image_p, self.config["num_octaves"], self.octave_scale)

        if self.depth:
            d_img = self.depth_model.inference(convert(image_p))
            d_img = d_img / np.max(d_img)

            if self.config["invert_depth"]:
                d_img = 1 - d_img

            if self.config["use_threshold"]:
                d_img[d_img < self.config["th_val"]] = 0

            d_img = np.expand_dims(d_img, 0)
            d_img = np.expand_dims(d_img, 0)
            d_img_octaves = get_octaves(
                d_img, self.config["num_octaves"], self.octave_scale
            )
            d_img_octaves = d_img_octaves[::-1]

        if mask is not None:
            mask = np.transpose(mask, (2, 0, 1))
            mask = np.expand_dims(mask, 0)
            octaves_mask = get_octaves(
                mask, self.config["num_octaves"], self.octave_scale
            )
            octaves_mask = octaves_mask[::-1]

        kernel = np.ones((5, 5), np.uint8)
        self.detail = np.zeros_like(octaves[-1])
        for octave, octave_base in enumerate(tqdm.tqdm(octaves[::-1], desc="Dreaming")):
            if octave > 0:
                # Upsample detail to new octave dimension
                self.detail = nd.zoom(
                    self.detail,
                    np.array(octave_base.shape) / np.array(self.detail.shape),
                    order=1,
                )

            input_image = octave_base + self.detail

            if mask is not None:
                dreamed_image = self.dream(
                    input_image, model, d_img_octaves[octave], octaves_mask[octave]
                )
            else:
                dreamed_image = self.dream(input_image, model, d_img_octaves[octave])

            self.detail = dreamed_image - octave_base

        return input_image

    def save_img(self, img, suffix, iter_):
        """Summary

        Args:
            img (numpy array): Output Image
            suffix (string): filename suffix
            iter_ (integer): the iteration value
        """
        img = deprocess(img)
        img = np.clip(img, 0, 1)
        file_name = self.img_list[self.config["start_position"] + iter_]
        file_name = file_name.split("/")[-1]
        plt.imsave(self.outpath + "/{}{}".format(suffix, file_name), img)

    def get_opflow_image(self, img1, dream_img, img2):
        """Calculates the optical flow with opencv and the spynet

        Args:
            img1 (TYPE): Description
            dream_img (TYPE): Description
            img2 (TYPE): Description

        Returns:
            TYPE: Description
        """
        img1 = np.float32(img1)
        dream_img = np.float32(dream_img)
        img2 = np.float32(img2)

        h, w, c = img1.shape
        if config["use_spynet"]:
            flow = calc_opflow(np.uint8(img1), np.uint8(img2))
            flow = np.transpose(np.float32(flow), (1, 2, 0))
        else:
            grayImg1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
            grayImg2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)

            flow = cv2.calcOpticalFlowFarneback(
                grayImg1,
                grayImg2,
                pyr_scale=0.5,
                levels=3,
                winsize=15,
                iterations=3,
                poly_n=3,
                poly_sigma=1.2,
                flags=0,
                flow=1,
            )

        inv_flow = flow
        flow = -flow

        flow[:, :, 0] += np.arange(w)
        flow[:, :, 1] += np.arange(h)[:, np.newaxis]

        halludiff = cv2.addWeighted(img2, 0.1, dream_img, 0.9, 0) - img1
        halludiff = cv2.remap(halludiff, flow, None, cv2.INTER_LINEAR)
        hallu_flow = img2 + halludiff

        magnitude, angle = cv2.cartToPolar(inv_flow[..., 0], inv_flow[..., 1])
        norm_mag = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
        ret, mask = cv2.threshold(norm_mag, 6, 255, cv2.THRESH_BINARY)
        flow_mask = mask.astype(np.uint8).reshape((h, w, 1))

        blendstatic = 0.1
        background_blendimg = cv2.addWeighted(
            img2, (1 - blendstatic), dream_img, blendstatic, 0
        )
        background_masked = cv2.bitwise_and(
            background_blendimg, background_blendimg, mask=cv2.bitwise_not(flow_mask)
        )

        return hallu_flow, background_masked

    def random_para(self):
        """chooses random parameters"""
        self.config["num_iterations"] = random.randint(2, 14)
        if not self.config["guided"]:
            self.at_layer_para = random.randint(10, 38)
        self.config["num_octaves"] = random.randint(30, 40)
        self.lr = random.choice([0.01, 0.009, 0.008, 0.02, 0.03, 0.007])
        self.octave_scale = random.choice(self.octave_list)

    def dream_single(self):
        """Dreams independent frames"""
        for i, path in enumerate(self.img_list):
            img1 = Image.open(path)
            d_img = self.deep_dream(self.transform(img1), self.model, i, seq="first")

            self.save_img(d_img, "", i)

    def dream_seq(self):
        """Dreams a sequence with optical flow"""

        for i, path in enumerate(self.img_list[self.config["start_position"] :]):

            if i == 0:
                img1 = Image.open(path)
                d_img = self.deep_dream(
                    self.transform(img1), self.model, i, seq="first"
                )

                self.save_img(d_img, "", i)
                d_img = convert(d_img)
                flow_iter = 0

                # the iterations needs to be reduced
                self.config["num_iterations"] -= 5

            if i > 0:
                img2 = Image.open(path)
                feature_img, background_masked = self.get_opflow_image(
                    img1, d_img, img2
                )

                feature_img = np.clip(feature_img, 0, 255)

                background_masked[background_masked > 0] = 1 - (flow_iter * 0.1)  # 0.5
                background_masked[background_masked == 0] = flow_iter * 0.1

                d_img = self.deep_dream(
                    self.transform(np.uint8(feature_img)),
                    self.model,
                    i,
                    seq="first",
                    mask=background_masked,
                )

                # change position
                img1 = img2
                self.save_img(d_img, "", i)
                d_img = convert(d_img)
                flow_iter += 1
                flow_iter = 0 if flow_iter > 5 else flow_iter


def start_dreamer(config):
    """

    Args:
        config (Dictionary): The config file
    """
    pretrained = config["pretrained"]

    # Load image
    img_p = config["input"] + "/*"
    outpath = config["outpath"]
    os.makedirs(outpath, exist_ok=True)

    dreamer = Dreamer(img_p, outpath, config)
    if config["seq"]:
        dreamer.dream_seq()
    else:
        dreamer.dream_single()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="", type=str)
    opt = parser.parse_args()

    config = load_config()

    if config["fp16"]:
        print("Imported Amp")
        from apex import amp

    start_dreamer(config)
