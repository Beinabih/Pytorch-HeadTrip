import os
import urllib.request

import cv2
import matplotlib.pyplot as plt
import torch


class MiDaS:
    def __init__(self, use_large_model):

        os.makedirs("depth_model", exist_ok=True)
        torch.hub.set_dir("depth_model")

        if use_large_model:
            self.midas = torch.hub.load(
                "intel-isl/MiDaS", "MiDaS", _use_new_zipfile_serialization=False
            )
        else:
            self.midas = torch.hub.load(
                "intel-isl/MiDaS", "MiDaS_small", _use_new_zipfile_serialization=False
            )

        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.midas.to(self.device)
        self.midas.eval()

        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

        if use_large_model:
            self.transform = midas_transforms.default_transform
        else:
            self.transform = midas_transforms.small_transform

    def inference(self, img):

        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        input_batch = self.transform(img).to(self.device)

        with torch.no_grad():
            prediction = self.midas(input_batch)

            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        return prediction.detach().cpu().numpy()
