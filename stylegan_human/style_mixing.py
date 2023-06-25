# Copyright (c) SenseTime Research. All rights reserved.

# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
# 


import os
import re
from typing import List
import legacy

import click
import dnnlib
import numpy as np
import PIL.Image
import torch

"""
Style mixing using pretrained network pickle.

Examples:

\b
python style_mixing.py --network=pretrained_models/stylegan_human_v2_1024.pkl --rows=85,100,75,458,1500 \\
    --cols=55,821,1789,293 --styles=0-3 --outdir=outputs/stylemixing 
"""

@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--rows', 'row_seeds', type=legacy.num_range, help='Random seeds to use for image rows', required=True)
@click.option('--cols', 'col_seeds', type=legacy.num_range, help='Random seeds to use for image columns', required=True)
@click.option('--styles', 'col_styles', type=legacy.num_range, help='Style layer range', default='0-6', show_default=True)
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=0.8, show_default=True)
@click.option('--noise-mode', help='Noise mode', type=click.Choice(['const', 'random', 'none']), default='const', show_default=True)
@click.option('--outdir', type=str, required=True, default='outputs/stylemixing')
def generate_style_mix(
    network_pkl: str,
    row_seeds: List[int],
    col_seeds: List[int],
    col_styles: List[int],
    truncation_psi: float,
    noise_mode: str,
    outdir: str
):
    
    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device)

    os.makedirs(outdir, exist_ok=True)

    print('Generating W vectors...')
    all_seeds = list(set(row_seeds + col_seeds))
    all_z = np.stack([np.random.RandomState(seed).randn(G.z_dim) for seed in all_seeds])
    all_w = G.mapping(torch.from_numpy(all_z).to(device), None)
    w_avg = G.mapping.w_avg
    all_w = w_avg + (all_w - w_avg) * truncation_psi
    w_dict = {seed: w for seed, w in zip(all_seeds, list(all_w))}

    print('Generating images...')
    all_images = G.synthesis(all_w, noise_mode=noise_mode)
    all_images = (all_images.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8).cpu().numpy()
    image_dict = {(seed, seed): image for seed, image in zip(all_seeds, list(all_images))}

    print('Generating style-mixed images...')
    for row_seed in row_seeds:
        for col_seed in col_seeds:
            w = w_dict[row_seed].clone()
            w[col_styles] = w_dict[col_seed][col_styles]
            image = G.synthesis(w[np.newaxis], noise_mode=noise_mode)
            image = (image.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            image_dict[(row_seed, col_seed)] = image[0].cpu().numpy()
    
    os.makedirs(outdir, exist_ok=True)
    # print('Saving images...')
    # for (row_seed, col_seed), image in image_dict.items():
    #     PIL.Image.fromarray(image, 'RGB').save(f'{outdir}/{row_seed}-{col_seed}.png')

    print('Saving image grid...')
    W = G.img_resolution // 2
    H = G.img_resolution 
    canvas = PIL.Image.new('RGB', (W * (len(col_seeds) + 1), H * (len(row_seeds) + 1)), 'black')
    for row_idx, row_seed in enumerate([0] + row_seeds):
        for col_idx, col_seed in enumerate([0] + col_seeds):
            if row_idx == 0 and col_idx == 0:
                continue
            key = (row_seed, col_seed)
            if row_idx == 0:
                key = (col_seed, col_seed)
            if col_idx == 0:
                key = (row_seed, row_seed)
            canvas.paste(PIL.Image.fromarray(image_dict[key], 'RGB'), (W * col_idx, H * row_idx))
    canvas.save(f'{outdir}/grid.png')


#----------------------------------------------------------------------------

if __name__ == "__main__":
    generate_style_mix() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
