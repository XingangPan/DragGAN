
# Copyright (c) SenseTime Research. All rights reserved.

"""Here we demo style-mixing results using StyleGAN2 pretrained model.
   Script reference: https://github.com/PDillis/stylegan2-fun """


import argparse
import legacy

import scipy
import numpy as np
import PIL.Image

import dnnlib
import dnnlib.tflib as tflib
from typing import List
import re
import sys
import os
import click
import torch

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import moviepy.editor


"""
Generate style mixing video. 
Examples:

\b
python stylemixing_video.py --network=pretrained_models/stylegan_human_v2_1024.pkl --row-seed=3859 \\
    --col-seeds=3098,31759,3791 --col-styles=8-12 --trunc=0.8 --outdir=outputs/stylemixing_video
"""

@click.command()
@click.option('--network', 'network_pkl', help='Path to network pickle filename', required=True)
@click.option('--row-seed', 'src_seed', type=legacy.num_range, help='Random seed to use for image source row', required=True)
@click.option('--col-seeds', 'dst_seeds', type=legacy.num_range, help='Random seeds to use for image columns (style)', required=True)
@click.option('--col-styles', 'col_styles', type=legacy.num_range, help='Style layer range (default: %(default)s)', default='0-6')
@click.option('--only-stylemix', 'only_stylemix', help='Add flag to only show the style mxied images in the video',default=False)
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi (default: %(default)s)', default=1)
@click.option('--duration-sec', 'duration_sec', type=float, help='Duration of video (default: %(default)s)', default=10)
@click.option('--fps', 'mp4_fps', type=int, help='FPS of generated video (default: %(default)s)', default=10)
@click.option('--indent-range', 'indent_range', type=int, default=30)
@click.option('--outdir', help='Root directory for run results (default: %(default)s)', default='outputs/stylemixing_video', metavar='DIR')

def style_mixing_video(network_pkl: str,
                       src_seed: List[int],                # Seed of the source image style (row)
                       dst_seeds: List[int],               # Seeds of the destination image styles (columns)
                       col_styles: List[int],              # Styles to transfer from first row to first column
                       truncation_psi=float,      
                       only_stylemix=bool,                 # True if user wishes to show only thre style transferred result
                       duration_sec=float,
                       smoothing_sec=1.0,
                       mp4_fps=int,
                       mp4_codec="libx264",
                       mp4_bitrate="16M",
                       minibatch_size=8,
                       noise_mode='const',
                       indent_range=int,
                       outdir=str):
    # Calculate the number of frames:
    print('col_seeds: ', dst_seeds)
    num_frames = int(np.rint(duration_sec * mp4_fps))
    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as f:
        Gs = legacy.load_network_pkl(f)['G_ema'].to(device) 

    print(Gs.num_ws, Gs.w_dim, Gs.img_resolution) 
    max_style = int(2 * np.log2(Gs.img_resolution)) - 3
    assert max(col_styles) <= max_style, f"Maximum col-style allowed: {max_style}"

    # Left col latents
    print('Generating Source W vectors...')
    src_shape = [num_frames] + [Gs.z_dim] 
    src_z = np.random.RandomState(*src_seed).randn(*src_shape).astype(np.float32) # [frames, src, component]
    src_z = scipy.ndimage.gaussian_filter(src_z, [smoothing_sec * mp4_fps] + [0] * (2- 1), mode="wrap") 
    src_z /= np.sqrt(np.mean(np.square(src_z)))
    # Map into the detangled latent space W and do truncation trick
    src_w = Gs.mapping(torch.from_numpy(src_z).to(device), None)
    w_avg = Gs.mapping.w_avg
    src_w = w_avg + (src_w - w_avg) * truncation_psi

    # Top row latents (fixed reference)
    print('Generating Destination W vectors...')  
    dst_z = np.stack([np.random.RandomState(seed).randn(Gs.z_dim) for seed in dst_seeds]) 
    dst_w = Gs.mapping(torch.from_numpy(dst_z).to(device), None)
    dst_w = w_avg + (dst_w - w_avg) * truncation_psi
    # Get the width and height of each image:
    H = Gs.img_resolution       # 1024
    W = Gs.img_resolution//2    # 512

    # Generate ALL the source images:
    src_images = Gs.synthesis(src_w, noise_mode=noise_mode)
    src_images = (src_images.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)

    # Generate the column images:
    dst_images = Gs.synthesis(dst_w, noise_mode=noise_mode)
    dst_images = (dst_images.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)


    print('Generating full video (including source and destination images)')
    # Generate our canvas where we will paste all the generated images:
    canvas = PIL.Image.new("RGB", ((W-indent_range) * (len(dst_seeds) + 1), H * (len(src_seed) + 1)), "white") # W, H

    for col, dst_image in enumerate(list(dst_images)): #dst_image:[3,1024,512]
        canvas.paste(PIL.Image.fromarray(dst_image.cpu().numpy(), "RGB"), ((col + 1) * (W-indent_range), 0)) #H
    # Aux functions: Frame generation func for moviepy.
    def make_frame(t):
        # Get the frame number according to time t:
        frame_idx = int(np.clip(np.round(t * mp4_fps), 0, num_frames - 1))
        # We wish the image belonging to the frame at time t:
        src_image = src_images[frame_idx] # always in the same place
        canvas.paste(PIL.Image.fromarray(src_image.cpu().numpy(), "RGB"), (0-indent_range, H)) # Paste it to the lower left

        # Now, for each of the column images:
        for col, dst_image in enumerate(list(dst_images)):
            # Select the pertinent latent w column:
            w_col = np.stack([dst_w[col].cpu()]) # [18, 512] -> [1, 18, 512]
            w_col = torch.from_numpy(w_col).to(device)
            # Replace the values defined by col_styles:
            w_col[:, col_styles] = src_w[frame_idx, col_styles]#.cpu()
            # Generate these synthesized images:
            col_images = Gs.synthesis(w_col, noise_mode=noise_mode)
            col_images = (col_images.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            # Paste them in their respective spot:
            for row, image in enumerate(list(col_images)):
                canvas.paste(
                    PIL.Image.fromarray(image.cpu().numpy(), "RGB"),
                    ((col + 1) * (W - indent_range), (row + 1) * H),
                )
        return np.array(canvas)
    
    # Generate video using make_frame:
    print('Generating style-mixed video...')
    videoclip = moviepy.editor.VideoClip(make_frame, duration=duration_sec)
    grid_size = [len(dst_seeds), len(src_seed)]
    mp4 = "{}x{}-style-mixing_{}_{}.mp4".format(*grid_size,min(col_styles),max(col_styles))
    if not os.path.exists(outdir): os.makedirs(outdir)
    videoclip.write_videofile(os.path.join(outdir,mp4),
                              fps=mp4_fps,
                              codec=mp4_codec,
                              bitrate=mp4_bitrate)


if __name__ == "__main__":
    style_mixing_video()



