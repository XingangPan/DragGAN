# StyleGAN-Human:  A Data-Centric Odyssey of Human Generation
<img src="./img/demo_V5_thumbnails-min.png" width="96%" height="96%">

<!--
**stylegan-human/StyleGAN-Human** is a ✨ _special_ ✨ repository because its `README.md` (this file) appears on your GitHub profile.

-->

> 
>
> **Abstract:** *Unconditional human image generation is an important task in vision and graphics, which enables various applications in the creative industry. Existing studies in this field mainly focus on "network engineering" such as designing new components and objective functions. This work takes a data-centric perspective and investigates multiple critical aspects in "data engineering", which we believe would complement the current practice. To facilitate a comprehensive study, we collect and annotate a large-scale human image dataset with over 230K samples capturing diverse poses and textures. Equipped with this large dataset, we rigorously investigate three essential factors in data engineering for StyleGAN-based human generation, namely data size, data distribution, and data alignment. Extensive experiments reveal several valuable observations w.r.t. these aspects: 1) Large-scale data, more than 40K images, are needed to train a high-fidelity unconditional human generation model with vanilla StyleGAN. 2) A balanced training set helps improve the generation quality with rare face poses compared to the long-tailed counterpart, whereas simply balancing the clothing texture distribution does not effectively bring an improvement. 3) Human GAN models with body centers for alignment outperform models trained using face centers or pelvis points as alignment anchors. In addition, a model zoo and human editing applications are demonstrated to facilitate future research in the community.* <br>
**Keyword:** Human Image Generation, Data-Centric, StyleGAN
 
[Jianglin Fu](mailto:fujianglin@sensetime.com), [Shikai Li](mailto:lishikai@sensetime.com), [Yuming Jiang](https://yumingj.github.io/), [Kwan-Yee Lin](https://kwanyeelin.github.io/), [Chen Qian](https://scholar.google.com/citations?user=AerkT0YAAAAJ&hl=zh-CN), [Chen Change Loy](https://www.mmlab-ntu.com/person/ccloy/), [Wayne Wu](https://wywu.github.io/), and [Ziwei Liu](https://liuziwei7.github.io/) <br>
**[[Demo Video]](https://youtu.be/nIrb9hwsdcI)** | **[[Project Page]](https://stylegan-human.github.io/)** | **[[Paper]](https://arxiv.org/pdf/2204.11823.pdf)**

## Updates
- [20/07/2022] [SHHQ-1.0](./docs/Dataset.md) dataset with 40K images is released! :sparkles:
- [15/06/2022] Data alignment and real-image inversion scripts are released. 
- [26/04/2022] Technical report released!
- [22/04/2022] Technical report will be released before May.
- [21/04/2022] The codebase and project page are created.

## Data Download
The first version SHHQ-1.0, with 40K images is released. To download and use the dataset set, please read the instructions in [Dataset.md](./docs/Dataset.md)

（We are currently facing large incoming applications, and we need to carefully verify all the applicants, please be patient, and we will reply to you as soon as possible.）

## Model Zoo

| Structure | 1024x512 | Metric | Scores |  512x256 | Metric | Scores | 
| --------- |:----------:| :----------:| :----------:|  :-----: |  :-----: |  :-----: | 
| StyleGAN1 |[stylegan_human_v1_1024.pkl](https://drive.google.com/file/d/1h-R-IV-INGdPEzj4P9ml6JTEvihuNgLX/view?usp=sharing)| fid50k | 3.79 | to be released | - | - |
| StyleGAN2 |[stylegan_human_v2_1024.pkl](https://drive.google.com/file/d/1FlAb1rYa0r_--Zj_ML8e6shmaF28hQb5/view?usp=sharing)| fid50k_full | 1.57 |[stylegan_human_v2_512.pkl](https://drive.google.com/file/d/1dlFEHbu-WzQWJl7nBBZYcTyo000H9hVm/view?usp=sharing) | fid50k_full | 1.97 |
| StyleGAN3 |to be released | - | - |  [stylegan_human_v3_512.pkl](https://drive.google.com/file/d/1_274jk_N6WSCkKWeu7hjHycqGvbuOFf5/view?usp=sharing) | fid50k_full | 2.54 |



## Web Demo 

Integrated into [Huggingface Spaces 🤗](https://huggingface.co/spaces) using [Gradio](https://github.com/gradio-app/gradio). Try out the Web Demo for generation: [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/hysts/StyleGAN-Human) and interpolation [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/hysts/StyleGAN-Human-Interpolation)



<a href="https://colab.research.google.com/drive/1sgxoDM55iM07FS54vz9ALg1XckiYA2On"><img src="https://colab.research.google.com/assets/colab-badge.svg" height=22.5></a> 

We prepare a Colab demo to allow you to synthesize images with the provided models, as well as visualize the performance of style-mixing, interpolation, and attributes editing.
The notebook will guide you to install the necessary environment and download pretrained models. The output images can be found in `./StyleGAN-Human/outputs/`.
Hope you enjoy!

## Usage

### System requirements
* The original code bases are [stylegan (tensorflow)](https://github.com/NVlabs/stylegan), [stylegan2-ada (pytorch)](https://github.com/NVlabs/stylegan2-ada-pytorch), [stylegan3 (pytorch)](https://github.com/NVlabs/stylegan3), released by NVidia

* We tested in Python 3.8.5 and PyTorch 1.9.1 with CUDA 11.1. (See https://pytorch.org for PyTorch install instructions.)

### Installation
To work with this project on your own machine, you need to install the environmnet as follows: 

```
conda env create -f environment.yml
conda activate stylehuman
# [Optional: tensorflow 1.x is required for StyleGAN1. ]
pip install nvidia-pyindex
pip install nvidia-tensorflow[horovod]
pip install nvidia-tensorboard==1.15
```
Extra notes:
1. In case having some conflicts when calling CUDA version, please try to empty the LD_LIBRARY_PATH. For example:
```
LD_LIBRARY_PATH=; python generate.py --outdir=out/stylegan_human_v2_1024 --trunc=1 --seeds=1,3,5,7 
--network=pretrained_models/stylegan_human_v2_1024.pkl --version 2
```


2. We found the following troubleshooting links might be helpful: [1.](https://github.com/NVlabs/stylegan3), [2.](https://github.com/NVlabs/stylegan3/blob/main/docs/troubleshooting.md)

### Train
The training scripts are based on the original [stylegan1](https://github.com/NVlabs/stylegan), [stylegan2-ada](https://github.com/NVlabs/stylegan2-ada-pytorch), and [stylegan3](https://github.com/NVlabs/stylegan3) with minor changes. Here we only provide the scripts with modifications for SG2 and SG3. You can replace the old files with the provided scripts to train. (assume SHHQ-1.0 is placed under data/)

#### Train Stylegan2-ada-pytorch with SHHQ-1.0
```
python train.py --outdir=training_results/sg2/ --data=data/SHHQ-1.0/ \
    --gpus=8 --aug=noaug --mirror=1 --snap=250 --cfg=shhq --square=False
```
#### Train Stylegan3 with SHHQ-1.0 
```
python train.py --outdir=training_results/sg3/ --cfg=stylegan3-r --gpus=8 --batch=32 --gamma=12.4 \
    --mirror=1 --aug=noaug --data=data/SHHQ-1.0/ --square=False --snap=250
```

### Pretrained models
Please put the downloaded pretrained models [from above link](#Model-Zoo) under the folder 'pretrained_models'.


### Generate full-body human images using our pretrained model
```
# Generate human full-body images without truncation
python generate.py --outdir=outputs/generate/stylegan_human_v2_1024 --trunc=1 --seeds=1,3,5,7 --network=pretrained_models/stylegan_human_v2_1024.pkl --version 2

# Generate human full-body images with truncation 
python generate.py --outdir=outputs/generate/stylegan_human_v2_1024 --trunc=0.8 --seeds=0-10 --network=pretrained_models/stylegan_human_v2_1024.pkl --version 2

# Generate human full-body images using stylegan V1
python generate.py --outdir=outputs/generate/stylegan_human_v1_1024 --network=pretrained_models/stylegan_human_v1_1024.pkl --version 1 --seeds=1,3,5

# Generate human full-body images using stylegan V3
python generate.py --outdir=outputs/generate/stylegan_human_v3_512 --network=pretrained_models/stylegan_human_v3_512.pkl --version 3 --seeds=1,3,5
```


#### Note: The following demos are generated based on models related to StyleGAN V2 (stylegan_human_v2_512.pkl and stylegan_human_v2_1024.pkl). If you want to see results for V1 or V3, you need to change the loading method of the corresponding models.


### Interpolation
```
python interpolation.py --network=pretrained_models/stylegan_human_v2_1024.pkl  --seeds=85,100 --outdir=outputs/inter_gifs
```

### Style-mixing **image** using stylegan2
```
python style_mixing.py --network=pretrained_models/stylegan_human_v2_1024.pkl --rows=85,100,75,458,1500 \\
    --cols=55,821,1789,293 --styles=0-3 --outdir=outputs/stylemixing 
```

### Style-mixing **video** using stylegan2
```
python stylemixing_video.py --network=pretrained_models/stylegan_human_v2_1024.pkl --row-seed=3859 \\
    --col-seeds=3098,31759,3791 --col-styles=8-12 --trunc=0.8 --outdir=outputs/stylemixing_video
```

### Aligned raw images
For alignment, we use [openpose-pytorch](https://github.com/Hzzone/pytorch-openpose) for body-keypoints detection and [PaddlePaddle](https://github.com/PaddlePaddle/PaddleSeg/tree/release/2.5/contrib/PP-HumanSeg) for human segmentation.
Before running the alignment script, few models need to be installed:
1. download [body_pose_model.pth](https://drive.google.com/drive/folders/1JsvI4M4ZTg98fmnCZLFM-3TeovnCRElG?usp=sharing) and place it into openpose/model/.
2. download and extract [deeplabv3p_resnet50_os8_humanseg_512x512_100k_with_softmax](https://paddleseg.bj.bcebos.com/dygraph/humanseg/export/deeplabv3p_resnet50_os8_humanseg_512x512_100k_with_softmax.zip) into PP_HumanSeg/export_model/deeplabv3p_resnet50_os8_humanseg_512x512_100k_with_softmax.
3. download and extract [deeplabv3p_resnet50_os8_humanseg_512x512_100k](https://paddleseg.bj.bcebos.com/dygraph/humanseg/train/deeplabv3p_resnet50_os8_humanseg_512x512_100k.zip) into PP_HumanSeg/pretrained_model/deeplabv3p_resnet50_os8_humanseg_512x512_100k.
4. install paddlepaddel: ``` pip install paddleseg ```

Then you can start alignment: 
```
python alignment.py --image-folder img/test/ --output-folder aligned_image/
```

### Invert real image with [PTI](https://github.com/danielroich/PTI)
Before inversion, please download our PTI weights: [e4e_w+.pt](https://drive.google.com/file/d/1NUfSJqLhsrU7c9PwAtlZ9xtrxhzS_6tu/view?usp=sharing) into /pti/.

Few parameters you can change:
- /pti/pti_configs/hyperparameters.py:
    - first_inv_type = 'w+' -> Use pretrained e4e encoder
    - first_inv_type = 'w'  -> Use projection and optimization
- /pti/pti_configs/paths_config.py:
    - input_data_path: path of real images
    - e4e: path of e4e_w+.pt
    - stylegan2_ada_shhq: pretrained stylegan2-ada model for SHHQ

```
python run_pti.py
```
Note: we used the test image under 'aligned_image/' (the output of alignment.py), the inverted latent code and fine-tuned generator will be saved in 'outputs/pti/'


### Editing with InterfaceGAN, StyleSpace, and Sefa
```
python edit.py --network pretrained_models/stylegan_human_v2_1024.pkl --attr_name upper_length \\
    --seeds 61531,61570,61571,61610 --outdir outputs/edit_results
``` 

### Editing using inverted latent code
```
python edit.py ---network outputs/pti/checkpoints/model_test.pkl --attr_name upper_length \\
    --outdir outputs/edit_results --real True --real_w_path outputs/pti/embeddings/test/PTI/test/0.pt --real_img_path aligned_image/test.png
```

Note: 
1. ''upper_length'' and ''bottom_length'' of ''attr_name'' are available for demo.
2. Layers to control and editing strength are set in edit/edit_config.py.


### Demo for [InsetGAN](https://arxiv.org/abs/2203.07293)

We implement a quick demo using the key idea from InsetGAN: combining the face generated by FFHQ with the human-body generated by our pretrained model, optimizing both face and body latent codes to get a coherent full-body image.
Before running the script, you need to download the [FFHQ face model]( https://docs.google.com/uc?export=download&confirm=t&id=125OG7SMkXI-Kf2aqiwLLHyCvSW-gZk3M), or you can use your own face model, as well as [pretrained face landmark](https://docs.google.com/uc?export=download&confirm=&id=1A82DnJBJzt8wI2J8ZrCK5fgHcQ2-tcWM) and [pretrained CNN face detection model for dlib](https://docs.google.com/uc?export=download&confirm=&id=1MduBgju5KFNrQfDLoQXJ_1_h5MnctCIG)
```
python insetgan.py --body_network=pretrained_models/stylegan_human_v2_1024.pkl --face_network=pretrained_models/ffhq.pkl \\
    --body_seed=82 --face_seed=43  --trunc=0.6 --outdir=outputs/insetgan/ --video 1 
```

## Results

### Editing with inverted real image
(from left to right: real image | inverted image | InterFaceGAN result | StyleSpace result | SeFa result)

https://user-images.githubusercontent.com/98547009/173773800-bb7fe54a-84d3-4b30-9864-a6b7b311f8ff.mp4


### For more demo, please visit our [**web page**](https://stylegan-human.github.io/) .
  

## TODO List

- [ ] Release 1024x512 version of StyleGAN-Human based on StyleGAN3 
- [ ] Release 512x256 version of StyleGAN-Human based on StyleGAN1 
- [ ] Extension of downstream application (InsetGAN): Add face inversion interface to support fusing user face image and stylegen-human body image
- [x] Add Inversion Script into the provided editing pipeline
- [ ] Release Dataset


## Related Works
* (SIGGRAPH 2022) **Text2Human: Text-Driven Controllable Human Image Generation**, Yuming Jiang et al. [[Paper](https://arxiv.org/pdf/2205.15996.pdf)], [[Code](https://github.com/yumingj/Text2Human)], [[Project Page](https://yumingj.github.io/projects/Text2Human.html)], [[Dataset](https://github.com/yumingj/DeepFashion-MultiModal)]
* (ICCV 2021) **Talk-to-Edit: Fine-Grained Facial Editing via Dialog**, Yuming Jiang et al. [[Paper](https://arxiv.org/abs/2109.04425)], [[Code](https://github.com/yumingj/Talk-to-Edit)], [[Project Page](https://www.mmlab-ntu.com/project/talkedit/)], [[Dataset](https://mmlab.ie.cuhk.edu.hk/projects/CelebA/CelebA_Dialog.html)]
* (Technical Report 2022) **Generalizable Neural Performer: Learning Robust Radiance Fields for Human Novel View Synthesis**, Wei Cheng et al. [[Paper](https://arxiv.org/pdf/2204.11798.pdf)], [[Code](https://github.com/generalizable-neural-performer/gnr)], [[Project Page](https://generalizable-neural-performer.github.io/)], [[Dataset](https://generalizable-neural-performer.github.io/genebody.html)]

## Citation

If you find this work useful for your research, please consider citing our paper:

```bibtex
@article{fu2022styleganhuman,
      title={StyleGAN-Human: A Data-Centric Odyssey of Human Generation}, 
      author={Fu, Jianglin and Li, Shikai and Jiang, Yuming and Lin, Kwan-Yee and Qian, Chen and Loy, Chen-Change and Wu, Wayne and Liu, Ziwei},
      journal   = {arXiv preprint},
      volume    = {arXiv:2204.11823},
      year    = {2022}
```

## Acknowlegement
Part of the code is borrowed from [stylegan (tensorflow)](https://github.com/NVlabs/stylegan), [stylegan2-ada (pytorch)](https://github.com/NVlabs/stylegan2-ada-pytorch), [stylegan3 (pytorch)](https://github.com/NVlabs/stylegan3).
