# Copyright (c) SenseTime Research. All rights reserved.

import os
import sys
import torch
import numpy as np
sys.path.append(".")
from torch_utils.models import Generator
import click
import cv2
from typing import List, Optional
import subprocess
import legacy
from edit.edit_helper import conv_warper, decoder, encoder_ifg, encoder_ss, encoder_sefa 


"""
Edit generated images with different SOTA methods. 
 Notes:
 1. We provide some latent directions in the folder, you can play around with them.
 2. ''upper_length'' and ''bottom_length'' of ''attr_name'' are available for demo.
 3. Layers to control and editing strength are set in edit/edit_config.py.
 
Examples:

\b
# Editing with InterfaceGAN, StyleSpace, and Sefa
python edit.py --network pretrained_models/stylegan_human_v2_1024.pkl --attr_name upper_length \\
    --seeds 61531,61570,61571,61610 --outdir outputs/edit_results


# Editing using inverted latent code
python edit.py ---network outputs/pti/checkpoints/model_test.pkl --attr_name upper_length  \\
    --outdir outputs/edit_results --real True --real_w_path outputs/pti/embeddings/test/PTI/test/0.pt --real_img_path aligned_image/test.png

"""



@click.command()
@click.pass_context
@click.option('--network', 'ckpt_path', help='Network pickle filename', required=True)
@click.option('--attr_name', help='choose one of the attr: upper_length or bottom_length', type=str, required=True)
@click.option('--trunc', 'truncation', type=float, help='Truncation psi', default=0.8, show_default=True)
@click.option('--gen_video', type=bool, default=True, help='If want to generate video')
@click.option('--combine', type=bool, default=True,  help='If want to combine different editing results in the same frame')
@click.option('--seeds', type=legacy.num_range, help='List of random seeds')
@click.option('--outdir', help='Where to save the output images', type=str, required=True, default='outputs/editing', metavar='DIR')
@click.option('--real', type=bool, help='True for editing real image', default=False)
@click.option('--real_w_path',  help='Path of latent code for real image')
@click.option('--real_img_path',  help='Path of real image, this just concat real image with inverted and edited results together')



def main(
    ctx: click.Context,
    ckpt_path: str,
    attr_name: str,
    truncation: float,
    gen_video: bool,
    combine: bool,
    seeds: Optional[List[int]],
    outdir: str,
    real: str,
    real_w_path: str,
    real_img_path: str
):
    ## convert pkl to pth
    # if not os.path.exists(ckpt_path.replace('.pkl','.pth')):
    legacy.convert(ckpt_path, ckpt_path.replace('.pkl','.pth'), G_only=real)
    ckpt_path = ckpt_path.replace('.pkl','.pth') 
    print("start...", flush=True)
    config = {"latent" : 512, "n_mlp" : 8, "channel_multiplier": 2}
    generator = Generator(
            size = 1024,
            style_dim=config["latent"],
            n_mlp=config["n_mlp"],
            channel_multiplier=config["channel_multiplier"]
        )
    
    generator.load_state_dict(torch.load(ckpt_path)['g_ema'])
    generator.eval().cuda()

    with torch.no_grad():
        mean_path = os.path.join('edit','mean_latent.pkl')
        if not os.path.exists(mean_path):
            mean_n = 3000
            mean_latent = generator.mean_latent(mean_n).detach()
            legacy.save_obj(mean_latent, mean_path)
        else:
            mean_latent = legacy.load_pkl(mean_path).cuda() 
        finals = []

        ## -- selected sample seeds -- ##
        # seeds = [60948,60965,61174,61210,61511,61598,61610] #bottom -> long
        #         [60941,61064,61103,61313,61531,61570,61571] # bottom -> short
        #         [60941,60965,61064,61103,6117461210,61531,61570,61571,61610] # upper --> long
        #         [60948,61313,61511,61598] # upper --> short
        if real: seeds = [0]

        for t in seeds:
            if real: # now assume process single real image only
                if real_img_path:
                    real_image = cv2.imread(real_img_path)
                    real_image = cv2.cvtColor(real_image, cv2.COLOR_BGR2RGB)
                    import torchvision.transforms as transforms
                    transform = transforms.Compose( # normalize to (-1, 1)
                        [transforms.ToTensor(),
                        transforms.Normalize(mean=(.5,.5,.5), std=(.5,.5,.5))]
                    )
                    real_image = transform(real_image).unsqueeze(0).cuda()                

                test_input = torch.load(real_w_path)
                output, _ = generator(test_input, False, truncation=1,input_is_latent=True, real=True)

            else: # generate image from random seeds
                test_input = torch.from_numpy(np.random.RandomState(t).randn(1, 512)).float().cuda()  # torch.Size([1, 512])
                output, _ = generator([test_input], False, truncation=truncation, truncation_latent=mean_latent, real=real)
            
            # interfacegan
            style_space, latent, noise = encoder_ifg(generator, test_input, attr_name, truncation, mean_latent,real=real)
            image1 = decoder(generator, style_space, latent, noise)
            # stylespace
            style_space, latent, noise = encoder_ss(generator, test_input, attr_name, truncation, mean_latent,real=real)
            image2 = decoder(generator, style_space, latent, noise)
            # sefa
            latent, noise = encoder_sefa(generator, test_input, attr_name, truncation, mean_latent,real=real)
            image3, _ = generator([latent], noise=noise, input_is_latent=True)
            if real_img_path:
                final = torch.cat((real_image, output, image1, image2, image3), 3)
            else:
                final = torch.cat((output, image1, image2, image3), 3)

            # legacy.visual(output, f'{outdir}/{attr_name}_{t:05d}_raw.jpg')
            # legacy.visual(image1, f'{outdir}/{attr_name}_{t:05d}_ifg.jpg')
            # legacy.visual(image2, f'{outdir}/{attr_name}_{t:05d}_ss.jpg')
            # legacy.visual(image3, f'{outdir}/{attr_name}_{t:05d}_sefa.jpg')

            if gen_video:
                total_step = 90
                if real:
                    video_ifg_path = f"{outdir}/video/ifg_{attr_name}_{real_w_path.split('/')[-2]}/"
                    video_ss_path = f"{outdir}/video/ss_{attr_name}_{real_w_path.split('/')[-2]}/"
                    video_sefa_path = f"{outdir}/video/ss_{attr_name}_{real_w_path.split('/')[-2]}/"
                else:
                    video_ifg_path = f"{outdir}/video/ifg_{attr_name}_{t:05d}/"
                    video_ss_path = f"{outdir}/video/ss_{attr_name}_{t:05d}/"
                    video_sefa_path = f"{outdir}/video/ss_{attr_name}_{t:05d}/"
                video_comb_path = f"{outdir}/video/tmp"    

                if combine:
                    if not os.path.exists(video_comb_path):
                        os.makedirs(video_comb_path)
                else:
                    if not os.path.exists(video_ifg_path):
                        os.makedirs(video_ifg_path)
                    if not os.path.exists(video_ss_path):
                        os.makedirs(video_ss_path)
                    if not os.path.exists(video_sefa_path):
                        os.makedirs(video_sefa_path)
                for i in range(total_step):
                    style_space, latent, noise = encoder_ifg(generator, test_input, attr_name, truncation, mean_latent, step=i, total=total_step,real=real)
                    image1 = decoder(generator, style_space, latent, noise)
                    style_space, latent, noise = encoder_ss(generator, test_input, attr_name, truncation, mean_latent, step=i, total=total_step,real=real)
                    image2 = decoder(generator, style_space, latent, noise)
                    latent, noise = encoder_sefa(generator, test_input, attr_name, truncation, mean_latent, step=i, total=total_step,real=real)
                    image3, _ = generator([latent], noise=noise, input_is_latent=True)
                    if combine:
                        if real_img_path:
                            comb_img = torch.cat((real_image, output, image1, image2, image3), 3)
                        else:
                            comb_img = torch.cat((output, image1, image2, image3), 3)
                        legacy.visual(comb_img, os.path.join(video_comb_path, f'{i:05d}.jpg'))
                    else:
                        legacy.visual(image1, os.path.join(video_ifg_path, f'{i:05d}.jpg'))
                        legacy.visual(image2, os.path.join(video_ss_path, f'{i:05d}.jpg'))
                if combine:
                    cmd=f"ffmpeg -hide_banner -loglevel error -y -r 30 -i {video_comb_path}/%05d.jpg -vcodec libx264 -pix_fmt yuv420p {video_ifg_path.replace('ifg_', '')[:-1] + '.mp4'}"
                    subprocess.call(cmd, shell=True)
                else:
                    cmd=f"ffmpeg -hide_banner -loglevel error -y -r 30 -i {video_ifg_path}/%05d.jpg -vcodec libx264 -pix_fmt yuv420p {video_ifg_path[:-1] + '.mp4'}"
                    subprocess.call(cmd, shell=True)
                    cmd=f"ffmpeg -hide_banner -loglevel error -y -r 30 -i {video_ss_path}/%05d.jpg -vcodec libx264 -pix_fmt yuv420p {video_ss_path[:-1] + '.mp4'}"
                    subprocess.call(cmd, shell=True)

        # interfacegan, stylespace, sefa
        finals.append(final)

        final = torch.cat(finals, 2)
        legacy.visual(final, os.path.join(outdir,'final.jpg'))


if __name__ == "__main__":
    main()