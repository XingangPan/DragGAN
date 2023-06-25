# Copyright (c) SenseTime Research. All rights reserved.

import torch
import cv2
from torchvision import transforms
import numpy as np
import math


def visual(output, out_path):
    output = (output + 1)/2
    output = torch.clamp(output, 0, 1)
    if output.shape[1] == 1:
        output = torch.cat([output, output, output], 1)
    output = output[0].detach().cpu().permute(1,2,0).numpy()
    output = (output*255).astype(np.uint8)
    output = output[:,:,::-1]
    cv2.imwrite(out_path, output)
    
    
def get_lr(t, initial_lr, rampdown=0.25, rampup=0.05):
    
    lr_ramp = min(1, (1 - t) / rampdown)
    lr_ramp = 0.5 - 0.5 * math.cos(lr_ramp * math.pi)
    lr_ramp = lr_ramp * min(1, t / rampup)
    return initial_lr * lr_ramp



def latent_noise(latent, strength):
    noise = torch.randn_like(latent) * strength

    return latent + noise

def noise_regularize_(noises):
    loss = 0

    for noise in noises:
        size = noise.shape[2]

        while True:
            loss = (
                loss
                + (noise * torch.roll(noise, shifts=1, dims=3)).mean().pow(2)
                + (noise * torch.roll(noise, shifts=1, dims=2)).mean().pow(2)
            )

            if size <= 8:
                break

            noise = noise.reshape([-1, 1, size // 2, 2, size // 2, 2])
            noise = noise.mean([3, 5])
            size //= 2

    return loss


def noise_normalize_(noises):
    for noise in noises:
        mean = noise.mean()
        std = noise.std()

        noise.data.add_(-mean).div_(std)
        

def tensor_to_numpy(x):
    x = x[0].permute(1, 2, 0)
    x = torch.clamp(x, -1 ,1)
    x = (x+1) * 127.5
    x = x.cpu().detach().numpy().astype(np.uint8)
    return x

def numpy_to_tensor(x):
    x = (x / 255 - 0.5) * 2
    x = torch.from_numpy(x).unsqueeze(0).permute(0, 3, 1, 2)
    x = x.cuda().float()
    return x

def tensor_to_pil(x):
    x = torch.clamp(x, -1 ,1)
    x = (x+1) * 127.5
    return transforms.ToPILImage()(x.squeeze_(0))

