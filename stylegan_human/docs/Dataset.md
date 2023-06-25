# SHHQ Dataset
<img src="../img/preview_samples1.png" width="96%" height="96%">

## Overview
SHHQ is a dataset with high-quality full-body human images in a resolution of 1024 × 512.
Since we need to follow a rigorous legal review in our institute, we can not release all of the data at once.

For now, SHHQ-1.0 with 40K images is released! More data will be released in the later versions.


## Data Sources
Images are collected in two main ways: 
1) From the Internet. 
We developed a crawler tool with an official API, mainly downloading images from Flickr, Pixabay and Pexels. So you need to meet all the following licenses when using the dataset: CC0, [Pixabay License](https://pixabay.com/service/license/), and [Pexels Licenses](https://www.pexels.com/license/).
2) From the data providers. 
We purchased images from databases of individual photographers, modeling agencies and other suppliers.
Images were reviewed by our legal team prior to purchase to ensure permission for use in research.

### Note: 
The composition of SHHQ-1.0: 

1) Images obtained from the above sources.
2) Processed 9991 DeepFashion [[1]](#1) images (retain only full body images).
3) 1940 African images from the InFashAI [[2]](#2) dataset to increase data diversity.

## Data License
We are aware of privacy concerns and seriously treat the license and privacy issues. All released data will be ensured under the license of CC0 and free for research use. Also, persons in the dataset are anonymised without additional private or sensitive metadata.

## Agreement
The SHHQ is available for non-commercial research purposes only. 

You agree not to reproduce, duplicate, copy, sell, trade, resell or exploit any portion of the images and any portion of the derived data for commercial purposes. 

You agree NOT to further copy, publish or distribute any portion of SHHQ to any third party for any purpose. Except, for internal use at a single site within the same organization it is allowed to make copies of the dataset.

Shanghai AI Lab reserves the right to terminate your access to the SHHQ at any time.

## Dataset Preview
For those interested in our dataset, we provide a preview version with 100 images randomly sampled from SHHQ-1.0: [SHHQ-1.0_samples](https://drive.google.com/file/d/1tnNFfmFtzRbYL3qEnNXQ_ShaN9YV5tI5/view?usp=sharing).

In SHHQ-1.0, we provide aligned raw images along with machine-calculated segmentation masks. Later we are planning to release manually annotated human-parsing version of these 40,000 images. Please stay tuned. 
 
> We also provide script [bg_white.py](../bg_white.py) to whiten the background of the raw image using its segmentation mask.

If you want to access the full SHHQ-1.0, please read the following instructions.

## Model trained using SHHQ-1.0

| Structure | 1024x512 | Metric | Scores |  512x256 | Metric | Scores | 
| --------- |:----------:| :----------:| :----------:|  :-----: |  :-----: |  :-----: | 
| StyleGAN1 | to be released | - | - | to be released | - | - |
| StyleGAN2 | [SHHQ-1.0_sg2_1024.pkl](https://drive.google.com/file/d/1PuvE72xpc69Zq4y58dohuKbG9dFnnjEX/view?usp=sharing) | fid50k_full | 3.56 | [SHHQ-1.0_sg2_512.pkl](https://drive.google.com/file/d/170t2FRWxR8_TG3_y0nVtDBogLPOClnyf/view?usp=sharing) | fid50k_full | 3.68 |
| StyleGAN3 | to be released | - | - |to be released | - | - |


## Download Instructions
Please download the SHHQ Dataset Release Agreement from [link](./SHHQ_Dataset_Release_Agreement.pdf).
Read it carefully, complete and sign it appropriately. 

Please send the completed form to Jianglin Fu (arlenefu@outlook.com) and Shikai Li (lishikai@pjlab.org.cn), and cc to Wayne Wu (wuwenyan0503@gmail.com) using institutional email address. The email Subject Title is "SHHQ Dataset Release Agreement". We will verify your request and contact you with the dataset link and password to unzip the image data.

Note:

1. We are currently facing large incoming applications, and we need to carefully verify all the applicants, please be patient, and we will reply to you as soon as possible.

2. The signature in the agreement should be hand-written.

## References
<a id="1">[1]</a> 
Liu, Ziwei and Luo, Ping and Qiu, Shi and Wang, Xiaogang and Tang, Xiaoou. DeepFashion: Powering Robust Clothes Recognition and Retrieval with Rich Annotations. CVPR (2016)

<a id="2">[2]</a> 
Hacheme, Gilles and Sayouti, Noureini. Neural fashion image captioning: Accounting for data diversity. arXiv preprint arXiv:2106.12154 (2021)

