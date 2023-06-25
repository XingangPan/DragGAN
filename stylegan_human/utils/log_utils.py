# Copyright (c) SenseTime Research. All rights reserved.


import numpy as np
from PIL import Image
import wandb
from pti.pti_configs import global_config
import torch
import matplotlib.pyplot as plt


def log_image_from_w(w, G, name):
    img = get_image_from_w(w, G)
    pillow_image = Image.fromarray(img)
    wandb.log(
        {f"{name}": [
            wandb.Image(pillow_image, caption=f"current inversion {name}")]},
        step=global_config.training_step)


def log_images_from_w(ws, G, names):
    for name, w in zip(names, ws):
        w = w.to(global_config.device)
        log_image_from_w(w, G, name)


def plot_image_from_w(w, G):
    img = get_image_from_w(w, G)
    pillow_image = Image.fromarray(img)
    plt.imshow(pillow_image)
    plt.show()


def plot_image(img):
    img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8).detach().cpu().numpy()
    pillow_image = Image.fromarray(img[0])
    plt.imshow(pillow_image)
    plt.show()


def save_image(name, method_type, results_dir, image, run_id):
    image.save(f'{results_dir}/{method_type}_{name}_{run_id}.jpg')


def save_w(w, G, name, method_type, results_dir):
    im = get_image_from_w(w, G)
    im = Image.fromarray(im, mode='RGB')
    save_image(name, method_type, results_dir, im)


def save_concat_image(base_dir, image_latents, new_inv_image_latent, new_G,
                      old_G,
                      file_name,
                      extra_image=None):
    images_to_save = []
    if extra_image is not None:
        images_to_save.append(extra_image)
    for latent in image_latents:
        images_to_save.append(get_image_from_w(latent, old_G))
    images_to_save.append(get_image_from_w(new_inv_image_latent, new_G))
    result_image = create_alongside_images(images_to_save)
    result_image.save(f'{base_dir}/{file_name}.jpg')


def save_single_image(base_dir, image_latent, G, file_name):
    image_to_save = get_image_from_w(image_latent, G)
    image_to_save = Image.fromarray(image_to_save, mode='RGB')
    image_to_save.save(f'{base_dir}/{file_name}.jpg')


def create_alongside_images(images):
    res = np.concatenate([np.array(image) for image in images], axis=1)
    return Image.fromarray(res, mode='RGB')


def get_image_from_w(w, G):
    if len(w.size()) <= 2:
        w = w.unsqueeze(0)
    with torch.no_grad():
        img = G.synthesis(w, noise_mode='const')
        img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8).detach().cpu().numpy()
    return img[0]
