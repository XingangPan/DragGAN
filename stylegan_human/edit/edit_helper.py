# Copyright (c) SenseTime Research. All rights reserved.

from legacy import save_obj, load_pkl
import torch
from torch.nn import functional as F
import pandas as pd
from .edit_config import attr_dict
import os

def conv_warper(layer, input, style, noise):
    # the conv should change
    conv = layer.conv
    batch, in_channel, height, width = input.shape

    style = style.view(batch, 1, in_channel, 1, 1)
    weight = conv.scale * conv.weight * style

    if conv.demodulate:
        demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + 1e-8)
        weight = weight * demod.view(batch, conv.out_channel, 1, 1, 1)

    weight = weight.view(
        batch * conv.out_channel, in_channel, conv.kernel_size, conv.kernel_size
    )

    if conv.upsample:
        input = input.view(1, batch * in_channel, height, width)
        weight = weight.view(
            batch, conv.out_channel, in_channel, conv.kernel_size, conv.kernel_size
        )
        weight = weight.transpose(1, 2).reshape(
            batch * in_channel, conv.out_channel, conv.kernel_size, conv.kernel_size
        )
        out = F.conv_transpose2d(input, weight, padding=0, stride=2, groups=batch)
        _, _, height, width = out.shape
        out = out.view(batch, conv.out_channel, height, width)
        out = conv.blur(out)

    elif conv.downsample:
        input = conv.blur(input)
        _, _, height, width = input.shape
        input = input.view(1, batch * in_channel, height, width)
        out = F.conv2d(input, weight, padding=0, stride=2, groups=batch)
        _, _, height, width = out.shape
        out = out.view(batch, conv.out_channel, height, width)

    else:
        input = input.view(1, batch * in_channel, height, width)
        out = F.conv2d(input, weight, padding=conv.padding, groups=batch)
        _, _, height, width = out.shape
        out = out.view(batch, conv.out_channel, height, width)
        
    out = layer.noise(out, noise=noise)
    out = layer.activate(out)
    
    return out

def decoder(G, style_space, latent, noise):
    # an decoder warper for G
    out = G.input(latent)
    out = conv_warper(G.conv1, out, style_space[0], noise[0])
    skip = G.to_rgb1(out, latent[:, 1])

    i = 1
    for conv1, conv2, noise1, noise2, to_rgb in zip(
        G.convs[::2], G.convs[1::2], noise[1::2], noise[2::2], G.to_rgbs
    ):
        out = conv_warper(conv1, out, style_space[i], noise=noise1)
        out = conv_warper(conv2, out, style_space[i+1], noise=noise2)
        skip = to_rgb(out, latent[:, i + 2], skip)
        i += 2
    image = skip

    return image

def encoder_ifg(G, noise, attr_name, truncation=1, truncation_latent=None, 
                  latent_dir='latent_direction/ss/',
                  step=0, total=0, real=False):
    if not real:
        styles = [noise]
        styles = [G.style(s) for s in styles]
    style_space = []
    
    if truncation<1:
        if not real: 
            style_t = []
            for style in styles:
                style_t.append(truncation_latent + truncation * (style - truncation_latent))
            styles = style_t
        else: # styles are latent (tensor: 1,18,512), for real PTI output
            truncation_latent = truncation_latent.repeat(18,1).unsqueeze(0) # (1,512) --> (1,18,512)
            styles = torch.add(truncation_latent,torch.mul(torch.sub(noise,truncation_latent),truncation))


    noise = [getattr(G.noises, 'noise_{}'.format(i)) for i in range(G.num_layers)]
    if not real:
        inject_index = G.n_latent
        latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)
    else: latent=styles

    style_space.append(G.conv1.conv.modulation(latent[:, 0]))
    i = 1
    for conv1, conv2, noise1, noise2, to_rgb in zip(
        G.convs[::2], G.convs[1::2], noise[1::2], noise[2::2], G.to_rgbs
    ):
        style_space.append(conv1.conv.modulation(latent[:, i]))
        style_space.append(conv2.conv.modulation(latent[:, i+1]))
        i += 2

    # get layer, strength by dict
    strength = attr_dict['interface_gan'][attr_name][0]

    if step != 0 and total != 0:
        strength = step / total * strength
    for i in range(15):
        style_vect = load_pkl(os.path.join(latent_dir, '{}/style_vect_mean_{}.pkl'.format(attr_name, i)))
        style_vect = torch.from_numpy(style_vect).to(latent.device).float()
        style_space[i] += style_vect * strength
        
    return style_space, latent, noise

def encoder_ss(G, noise, attr_name, truncation=1, truncation_latent=None, 
               statics_dir="latent_direction/ss_statics",
               latent_dir="latent_direction/ss/",
               step=0, total=0,real=False):
    if not real:
        styles = [noise]
        styles = [G.style(s) for s in styles]
    style_space = []

    if truncation<1:
        if not real:
            style_t = []
            for style in styles:
                style_t.append(
                    truncation_latent + truncation * (style - truncation_latent)
                )
            styles = style_t
        else: # styles are latent (tensor: 1,18,512), for real PTI output
            truncation_latent = truncation_latent.repeat(18,1).unsqueeze(0) # (1,512) --> (1,18,512)
            styles = torch.add(truncation_latent,torch.mul(torch.sub(noise,truncation_latent),truncation))

    noise = [getattr(G.noises, 'noise_{}'.format(i)) for i in range(G.num_layers)]
    
    if not real:
        inject_index = G.n_latent
        latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)
    else: latent = styles

    style_space.append(G.conv1.conv.modulation(latent[:, 0]))
    i = 1
    for conv1, conv2, noise1, noise2, to_rgb in zip(
        G.convs[::2], G.convs[1::2], noise[1::2], noise[2::2], G.to_rgbs
    ):
        style_space.append(conv1.conv.modulation(latent[:, i]))
        style_space.append(conv2.conv.modulation(latent[:, i+1]))
        i += 2
    # get threshold, layer, strength by dict
    layer, strength, threshold = attr_dict['stylespace'][attr_name] 

    statis_dir = os.path.join(statics_dir, "{}_statis/{}".format(attr_name, layer))
    statis_csv_path = os.path.join(statis_dir, "statis.csv")
    statis_df = pd.read_csv(statis_csv_path)
    statis_df = statis_df.sort_values(by='channel', ascending=True)
    ch_mask = statis_df['strength'].values
    ch_mask = torch.from_numpy(ch_mask).to(latent.device).float()
    ch_mask = (ch_mask.abs()>threshold).float()
    style_vect = load_pkl(os.path.join(latent_dir, '{}/style_vect_mean_{}.pkl'.format(attr_name, layer)))
    style_vect = torch.from_numpy(style_vect).to(latent.device).float()

    style_vect = style_vect * ch_mask

    if step != 0 and total != 0:
        strength = step / total * strength

    style_space[layer] += style_vect * strength
        
    return style_space, latent, noise

def encoder_sefa(G, noise, attr_name, truncation=1, truncation_latent=None, 
                  latent_dir='latent_direction/sefa/',
                  step=0, total=0, real=False):
    if not real: 
        styles = [noise]
        styles = [G.style(s) for s in styles]
    
    if truncation<1:
        if not real:
            style_t = []
            for style in styles:
                style_t.append(
                    truncation_latent + truncation * (style - truncation_latent)
                )
            styles = style_t
        else:
            truncation_latent = truncation_latent.repeat(18,1).unsqueeze(0) # (1,512) --> (1,18,512)
            styles = torch.add(truncation_latent,torch.mul(torch.sub(noise,truncation_latent),truncation))


    noise = [getattr(G.noises, 'noise_{}'.format(i)) for i in range(G.num_layers)]
    if not real:
        inject_index = G.n_latent
        latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)
    else: latent = styles
    
    layer, strength = attr_dict['sefa'][attr_name] 

    sefa_vect = torch.load(os.path.join(latent_dir, '{}.pt'.format(attr_name))).to(latent.device).float()
    if step != 0 and total != 0:
        strength = step / total * strength
    for l in layer:
        latent[:, l, :] += (sefa_vect * strength * 2)
        

    return latent, noise
