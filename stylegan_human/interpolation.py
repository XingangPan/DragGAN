# Copyright (c) SenseTime Research. All rights reserved.

## interpolate between two z code
## score all middle latent code
# https://www.aiuai.cn/aifarm1929.html

import os
import re
from typing import List
from tqdm import tqdm
import click
import dnnlib
import numpy as np
import PIL.Image
import torch
import click
import legacy
import random
from typing import List, Optional


def lerp(code1, code2, alpha):
    return code1 * alpha + code2 * (1 - alpha)

# Taken and adapted from wikipedia's slerp article
# https://en.wikipedia.org/wiki/Slerp
def slerp(code1, code2, alpha, DOT_THRESHOLD=0.9995): # Spherical linear interpolation
    code1_copy = np.copy(code1)
    code2_copy = np.copy(code2)

    code1 = code1 / np.linalg.norm(code1)
    code2 = code2 / np.linalg.norm(code2)
    dot = np.sum(code1 * code2)
    if np.abs(dot) > DOT_THRESHOLD:
        return lerp(code1_copy, code2_copy, alpha)

    # Calculate initial angle between v0 and v1
    theta_0 = np.arccos(dot)
    sin_theta_0 = np.sin(theta_0)
    # Angle at timestep t
    theta_t = theta_0 * alpha
    sin_theta_t = np.sin(theta_t)

    s0 = np.sin(theta_0 - theta_t) / sin_theta_0
    s1 = sin_theta_t / sin_theta_0
    code3 = s0 * code1_copy + s1 * code2_copy
    return code3

def generate_image_from_z(G, z, noise_mode, truncation_psi, device):
    label = torch.zeros([1, G.c_dim], device=device)
    w = G.mapping(z, label,truncation_psi=truncation_psi)
    img = G.synthesis(w, noise_mode=noise_mode,force_fp32 = True)
    img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    img = PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB')
    return img


def get_concat_h(im1, im2):
    dst = PIL.Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst


def make_latent_interp_animation(G, code1, code2, img1, img2, num_interps, noise_mode, save_mid_image, truncation_psi,device, outdir,fps):
    step_size = 1.0/num_interps
    
    all_imgs = []
    amounts = np.arange(0, 1, step_size)
    for seed_idx, alpha in enumerate(tqdm(amounts)):
        interpolated_latent_code = lerp(code1, code2, alpha)
        image = generate_image_from_z(G,interpolated_latent_code, noise_mode, truncation_psi, device)
        interp_latent_image = image.resize((512, 1024))
        if not os.path.exists(os.path.join(outdir,'img')): os.makedirs(os.path.join(outdir,'img'), exist_ok=True)
        if save_mid_image:  
            interp_latent_image.save(f'{outdir}/img/seed{seed_idx:04d}.png')

        frame = get_concat_h(img2, interp_latent_image)
        frame = get_concat_h(frame, img1)
        all_imgs.append(frame)

    save_name = os.path.join(outdir,'latent_space_traversal.gif')
    all_imgs[0].save(save_name, save_all=True, append_images=all_imgs[1:], duration=1000/fps, loop=0)
    

"""
Create interpolated images between two given seeds using pretrained network pickle.

Examples:

\b
python interpolation.py --network=pretrained_models/stylegan_human_v2_1024.pkl  --seeds=85,100 --outdir=outputs/inter_gifs
    
"""

@click.command()
@click.pass_context
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--seeds', type=legacy.num_range, help='List of 2 random seeds, e.g. 1,2')
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=0.8, show_default=True)
@click.option('--noise-mode', 'noise_mode', help='Noise mode', type=click.Choice(['const', 'random', 'none']), default='const', show_default=True)
@click.option('--outdir', default= 'outputs/inter_gifs', help='Where to save the output images', type=str, required=True, metavar='DIR')
@click.option('--save_mid_image', default=True, type=bool, help='select True if you want to save all interpolated images')
@click.option('--fps', default= 15, help='FPS for GIF', type=int)
@click.option('--num_interps', default= 100, help='Number of interpolation images', type=int)
def main(
    ctx: click.Context,
    network_pkl: str,
    seeds: Optional[List[int]],
    truncation_psi: float,
    noise_mode: str,
    outdir: str,
    save_mid_image: bool,
    fps:int,
    num_interps:int
):
    

    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore

    outdir = os.path.join(outdir)
    if not os.path.exists(outdir):
        os.makedirs(outdir,exist_ok=True)
        os.makedirs(os.path.join(outdir,'img'),exist_ok=True)

    if len(seeds) > 2: 
        print("Receiving more than two seeds, only use the first two.")
        seeds = seeds[0:2]
    elif len(seeds) == 1: 
        print('Require two seeds, randomly generate two now.')
        seeds = [seeds[0],random.randint(0,10000)]

    z1 = torch.from_numpy(np.random.RandomState(seeds[0]).randn(1, G.z_dim)).to(device)
    z2 = torch.from_numpy(np.random.RandomState(seeds[1]).randn(1, G.z_dim)).to(device)
    img1 = generate_image_from_z(G, z1, noise_mode, truncation_psi, device)
    img2 = generate_image_from_z(G, z2, noise_mode, truncation_psi, device)
    img1.save(f'{outdir}/seed{seeds[0]:04d}.png')
    img2.save(f'{outdir}/seed{seeds[1]:04d}.png')

    make_latent_interp_animation(G, z1, z2, img1, img2, num_interps, noise_mode, save_mid_image, truncation_psi, device, outdir, fps)    


if __name__ == "__main__":
    main() 
