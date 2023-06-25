# Copyright (c) SenseTime Research. All rights reserved.

import os
import click
import cv2
import numpy as np

def bg_white(seg, raw, blur_level=3, gaussian=81):
    seg = cv2.blur(seg, (blur_level, blur_level))

    empty = np.ones_like(seg)
    seg_bg = (empty - seg) * 255 
    seg_bg = cv2.GaussianBlur(seg_bg,(gaussian,gaussian),0)

    background_mask = cv2.cvtColor(255 - cv2.cvtColor(seg, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    masked_fg = (raw * (1 / 255)) * (seg * (1 / 255))
    masked_bg = (seg_bg * (1 / 255)) * (background_mask * (1 / 255))

    frame = np.uint8(cv2.add(masked_bg,masked_fg)*255)

    return frame


"""
To turn background into white.

Examples:

\b
python bg_white.py  --raw_img_dir=./SHHQ-1.0/no_segment/ --raw_seg_dir=./SHHQ-1.0/segments/ \\
    --outdir=./SHHQ-1.0/bg_white/
"""

@click.command()
@click.pass_context
@click.option('--raw_img_dir', default="./SHHQ-1.0/no_segment/", help='folder of raw image', required=True)
@click.option('--raw_seg_dir', default='./SHHQ-1.0/segments/', help='folder of segmentation masks', required=True)
@click.option('--outdir', help='Where to save the output images', default= "./SHHQ-1.0/bg_white/" , type=str, required=True, metavar='DIR')

def main(
        ctx: click.Context,
        raw_img_dir: str,
        raw_seg_dir: str,
        outdir: str):
    os.makedirs(outdir, exist_ok=True)
    files = os.listdir(raw_img_dir)
    for file in files: 
        print(file)
        raw = cv2.imread(os.path.join(raw_img_dir, file))
        seg = cv2.imread(os.path.join(raw_seg_dir, file))
        assert raw is not None
        assert seg is not None
        white_frame = bg_white(seg, raw)
        cv2.imwrite(os.path.join(outdir,file), white_frame)

if __name__ == "__main__":
    main()