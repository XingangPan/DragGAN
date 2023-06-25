# Copyright (c) SenseTime Research. All rights reserved.

# Copyright (c) 2019, NVIDIA Corporation. All rights reserved.
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://nvlabs.github.io/stylegan2/license.html


## this script is for generating images from pre-trained network based on StyleGAN1 (TensorFlow) and StyleGAN2-ada (PyTorch) ##

import os
import click
import dnnlib
import numpy as np
import PIL.Image
import legacy
from typing import List, Optional

"""
Generate images using pretrained network pickle.
Examples:

\b
# Generate human full-body images without truncation
python generate.py --outdir=outputs/generate/stylegan_human_v2_1024 --trunc=1 --seeds=1,3,5,7 \\
    --network=pretrained_models/stylegan_human_v2_1024.pkl --version 2

\b
# Generate human full-body images with truncation 
python generate.py --outdir=outputs/generate/stylegan_human_v2_1024 --trunc=0.8 --seeds=0-100\\
    --network=pretrained_models/stylegan_human_v2_1024.pkl --version 2

# \b
# Generate human full-body images using stylegan V1
# python generate.py --outdir=outputs/generate/stylegan_human_v1_1024  \\
#     --network=pretrained_models/stylegan_human_v1_1024.pkl --version 1
"""


@click.command()
@click.pass_context
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--seeds', type=legacy.num_range, help='List of random seeds')
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=1, show_default=True)
@click.option('--noise-mode', help='Noise mode', type=click.Choice(['const', 'random', 'none']), default='const', show_default=True)
@click.option('--outdir', help='Where to save the output images', default= 'outputs/generate/' , type=str, required=True, metavar='DIR')
@click.option('--version', help="stylegan version, 1, 2 or 3", type=int, default=2)
def generate_images(
    ctx: click.Context,
    network_pkl: str,
    seeds: Optional[List[int]],
    truncation_psi: float,
    noise_mode: str,
    outdir: str,
    version: int
):

    print('Loading networks from "%s"...' % network_pkl)
    if version == 1: 
        import dnnlib.tflib as tflib
        tflib.init_tf()
        G, D, Gs = legacy.load_pkl(network_pkl)

    else: 
        import torch
        device = torch.device('cuda')
        with dnnlib.util.open_url(network_pkl) as f:
            G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore
    os.makedirs(outdir, exist_ok=True)


    if seeds is None:
        ctx.fail('--seeds option is required.')


    # Generate images.
    target_z = np.array([])
    target_w = np.array([])
    latent_out = outdir.replace('/images/','')
    for seed_idx, seed in enumerate(seeds):
        if seed % 5000 == 0:
            print('Generating image for seed %d (%d/%d) ...' % (seed, seed_idx, len(seeds)))

        if version == 1: ## stylegan v1
            z =  np.random.RandomState(seed).randn(1, Gs.input_shape[1])     
            # Generate image.
            fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
            if noise_mode == 'const': randomize_noise=False
            else: randomize_noise = True
            images = Gs.run(z, None, truncation_psi=truncation_psi, randomize_noise=randomize_noise, output_transform=fmt)
            PIL.Image.fromarray(images[0], 'RGB').save(f'{outdir}/seed{seed:04d}.png')

        else: ## stylegan v2/v3
            label = torch.zeros([1, G.c_dim], device=device)
            z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)
            if target_z.size==0:
                target_z= z.cpu()
            else:
                target_z=np.append(target_z, z.cpu(), axis=0) 

            w = G.mapping(z, label,truncation_psi=truncation_psi)
            img = G.synthesis(w, noise_mode=noise_mode,force_fp32 = True)
            if target_w.size==0:
                target_w= w.cpu()
            else:
                target_w=np.append(target_w, w.cpu(), axis=0) 

            img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(f'{outdir}/seed{seed:04d}.png')
    # print(target_z)
    # print(target_z.shape,target_w.shape)


#----------------------------------------------------------------------------

if __name__ == "__main__":
    generate_images() 

#----------------------------------------------------------------------------