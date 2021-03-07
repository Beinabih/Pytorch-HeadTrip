import math
import numbers

import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as nd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms

mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

preprocess = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize(mean, std)]
)


def show_img(img):
    img = convert(img.detach().cpu().numpy())
    plt.imshow(img)
    plt.show()


def warp(x, flo):
    """
    warp an image/tensor (im2) back to im1, according to the optical flow
    x: [B, C, H, W] (im2)
    flo: [B, 2, H, W] flow
    """
    B, C, H, W = x.size()
    # mesh grid
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).float()

    if x.is_cuda:
        grid = grid.cuda()
        flo = flo.cuda()
    vgrid = Variable(grid) + flo

    # scale grid to [-1,1]
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :] / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :] / max(H - 1, 1) - 1.0

    vgrid = vgrid.permute(0, 2, 3, 1)
    output = nn.functional.grid_sample(x, vgrid)
    mask = torch.autograd.Variable(torch.ones(x.size())).cuda()
    mask = nn.functional.grid_sample(mask, vgrid)

    # if W==128:
    # np.save('mask.npy', mask.cpu().data.numpy())
    # np.save('warp.npy', output.cpu().data.numpy())

    mask[mask < 0.9999] = 0
    mask[mask > 0] = 1

    return output * mask, mask


def get_octaves(img, num_octaves, octave_scale):
    octaves = [img]
    for _ in range(num_octaves - 1):
        new_octave = nd.zoom(
            octaves[-1], (1, 1, 1 / octave_scale, 1 / octave_scale), order=1
        )

        if new_octave.shape[2] > 32 and new_octave.shape[3] > 32:
            octaves.append(new_octave)

    return octaves


def deprocess(image_np):
    image_np = image_np.squeeze().transpose(1, 2, 0)
    image_np = image_np * std.reshape((1, 1, 3)) + mean.reshape((1, 1, 3))
    image_np = np.clip(image_np, 0.0, 255.0)
    return image_np


def clip(image_tensor):
    for c in range(3):
        m, s = mean[c], std[c]
        image_tensor[0, c] = torch.clamp(image_tensor[0, c], -m / s, (1 - m) / s)
    return image_tensor


def convert(img):
    image_np = img.squeeze().transpose(1, 2, 0)
    image_np = image_np * std.reshape((1, 1, 3)) + mean.reshape((1, 1, 3))
    image_np = image_np * 255
    image_np = np.clip(image_np, 0.0, 255.0)
    return image_np.astype(np.uint8)


def find_contours(img):
    img = convert(img)
    imgray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    ret, thresh = cv2.threshold(imgray, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    # for contour in contours:
    #     hull = cv2.convexHull(contour)
    #     cv2.drawContours(imgrey,[hull],-1,0,-1)
    if contours:
        area = []
        for cnt in contours:
            area.append(cv2.contourArea(cnt))

        hull = []
        arr = np.array(area)
        sorted_arr = np.flip((np.argsort(arr)))

        mask = np.zeros(img.shape)

        if len(contours) == 1:
            hull.append(cv2.convexHull(contours[sorted_arr[0]]))
            # img = cv2.drawContours(mask, [hull[0]], 0, (255,255,255), 3)
            img = cv2.fillPoly(img, pts=[hull], color=(255, 255, 255))
        elif len(contours) == 2:
            for i in range(2):
                hull.append(cv2.convexHull(contours[sorted_arr[i]]))
            img = cv2.drawContours(mask, [hull[0]], 0, (255, 255, 255), 3)
            img = cv2.drawContours(mask, [hull[1]], 0, (255, 255, 255), 3)
            img = cv2.fillPoly(img, pts=[hull], color=(255, 255, 255))
        elif len(contours) == 3:
            for i in range(3):
                hull.append(cv2.convexHull(contours[sorted_arr[i]]))
            img = cv2.drawContours(mask, [hull[0]], 0, (255, 255, 255), 3)
            img = cv2.drawContours(mask, [hull[1]], 0, (255, 255, 255), 3)
            img = cv2.drawContours(mask, [hull[2]], 0, (255, 255, 255), 3)
            img = cv2.fillPoly(img, pts=[hull], color=(255, 255, 255))
        elif len(contours) >= 4:
            for i in range(4):
                hull.append(cv2.convexHull(contours[sorted_arr[i]]))

            for i in range(4):
                img = cv2.drawContours(mask, [hull[i]], 0, (255, 255, 255), 3)
                img = cv2.fillPoly(img, pts=[hull[i]], color=(255, 255, 255))
                # img = cv2.drawContours(mask, [contours[sorted_arr[1]]], 0, (255,255,255), 3)
                # img = cv2.drawContours(mask, [contours[sorted_arr[2]]], 0, (255,255,255), 3)
                # img = cv2.drawContours(mask, [contours[sorted_arr[3]]], 0, (255,255,255), 3)

        # img = cv2.drawContours(img, [hull[0]], -1, (0,255,0), -1)
        # cv2.imshow('imgcont', img)
        # cv2.waitKey(0)

    # img = img.transpose(2, 0, 1)
    # img = np.expand_dims(img, 0)
    return img[:, :, 0]


def make_video(path, name):

    image = cv2.imread(path[0])
    height, width, layers = image.shape
    size = (width, height)

    out = cv2.VideoWriter(
        name,
        cv2.VideoWriter_fourcc(*"MP4V"),
        framerate,
        size,
    )

    for i in range(65):
        img_array = []
        for filename in path:
            image = cv2.imread(filename)
            img_array.append(image)

        for i in range(len(img_array)):
            out.write(img_array[i])
    out.release()


def blend(self, img1, img2, blend):
    return img1 * (1.0 - blend) + img2 * blend


def blend_intermediate(self):
    for i in range(1, self.batchsize - 1):
        n = i * 1 / (self.batchsize - 1)
        blend_grad = self.blend(self.detail_first, self.detail_last, n)

        if i > 1:
            self.blend_gradients = np.concatenate((self.blend_gradients, blend_grad), 0)
        else:
            self.blend_gradients = blend_grad

    # if self.config['seq'] and i > 0:
    # ###optical flow
    # flow = calc_opflow(self.prev_img, unp_image)
    # flow = -flow
    # warped_out = warp(self.prev_out[octave], flow)
    # loss = -(self.loss(out, target) + self.l1(out, warped_out))


def smooth_grad(grad, octave):
    if octave < 5:
        smoothing = GaussianSmoothing(3, 7, 10)
        inp = F.pad(grad, (3, 3, 3, 3), mode="reflect")
        return smoothing(inp)
    else:
        return grad


class GaussianSmoothing(nn.Module):
    """
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """

    def __init__(self, channels, kernel_size, sigma, dim=2):
        super(GaussianSmoothing, self).__init__()
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [torch.arange(size, dtype=torch.float32) for size in kernel_size]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= (
                1
                / (std * math.sqrt(2 * math.pi))
                * torch.exp(-(((mgrid - mean) / std) ** 2) / 2)
            )

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer("weight", kernel)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                "Only 1, 2 and 3 dimensions are supported. Received {}.".format(dim)
            )

    def forward(self, input):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        return self.conv(input, weight=self.weight.cuda(), groups=self.groups)
