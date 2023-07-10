<p align="center">

  <h1 align="center">Drag Your GAN: Interactive Point-based Manipulation on the Generative Image Manifold</h1>
  <p align="center">
    <a href="https://xingangpan.github.io/"><strong>Xingang Pan</strong></a>
    ·
    <a href="https://ayushtewari.com/"><strong>Ayush Tewari</strong></a>
    ·
    <a href="https://people.mpi-inf.mpg.de/~tleimkue/"><strong>Thomas Leimkühler</strong></a>
    ·
    <a href="https://lingjie0206.github.io/"><strong>Lingjie Liu</strong></a>
    ·
    <a href="https://www.meka.page/"><strong>Abhimitra Meka</strong></a>
    ·
    <a href="http://www.mpi-inf.mpg.de/~theobalt/"><strong>Christian Theobalt</strong></a>
  </p>
  <h2 align="center">SIGGRAPH 2023 Conference Proceedings</h2>
  <div align="center">
    <img src="DragGAN.gif", width="600">
  </div>

  <p align="center">
  <br>
    <a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
    <a href="https://twitter.com/XingangP"><img alt='Twitter' src="https://img.shields.io/twitter/follow/XingangP?label=%40XingangP"></a>
    <a href="https://arxiv.org/abs/2305.10973">
      <img src='https://img.shields.io/badge/Paper-PDF-green?style=for-the-badge&logo=adobeacrobatreader&logoWidth=20&logoColor=white&labelColor=66cc00&color=94DD15' alt='Paper PDF'>
    </a>
    <a href='https://vcai.mpi-inf.mpg.de/projects/DragGAN/'>
      <img src='https://img.shields.io/badge/DragGAN-Page-orange?style=for-the-badge&logo=Google%20chrome&logoColor=white&labelColor=D35400' alt='Project Page'></a>
    <a href="https://colab.research.google.com/drive/1mey-IXPwQC_qSthI5hO-LTX7QL4ivtPh?usp=sharing"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>
  </p>
</p>

## Web Demos

[![Open in OpenXLab](https://cdn-static.openxlab.org.cn/app-center/openxlab_app.svg)](https://openxlab.org.cn/apps/detail/XingangPan/DragGAN)

<p align="left">
  <a href="https://huggingface.co/spaces/radames/DragGan"><img alt="Huggingface" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-DragGAN-orange"></a>
</p>

## Requirements

If you have CUDA graphic card, please follow the requirements of [NVlabs/stylegan3](https://github.com/NVlabs/stylegan3#requirements).  

The usual installation steps involve the following commands, they should set up the correct CUDA version and all the python packages

```
conda env create -f environment.yml
conda activate stylegan3
```

Then install the additional requirements

```
pip install -r requirements.txt
```

Otherwise (for GPU acceleration on MacOS with Silicon Mac M1/M2, or just CPU) try the following:

```sh
cat environment.yml | \
  grep -v -E 'nvidia|cuda' > environment-no-nvidia.yml && \
    conda env create -f environment-no-nvidia.yml
conda activate stylegan3

# On MacOS
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

## Run Gradio visualizer in Docker 

Provided docker image is based on NGC PyTorch repository. To quickly try out visualizer in Docker, run the following:  

```sh
# before you build the docker container, make sure you have cloned this repo, and downloaded the pretrained model by `python scripts/download_model.py`.
docker build . -t draggan:latest  
docker run -p 7860:7860 -v "$PWD":/workspace/src -it draggan:latest bash
# (Use GPU)if you want to utilize your Nvidia gpu to accelerate in docker, please add command tag `--gpus all`, like:
#   docker run --gpus all  -p 7860:7860 -v "$PWD":/workspace/src -it draggan:latest bash

cd src && python visualizer_drag_gradio.py --listen
```
Now you can open a shared link from Gradio (printed in the terminal console).   
Beware the Docker image takes about 25GB of disk space!

## Download pre-trained StyleGAN2 weights

To download pre-trained weights, simply run:

```
python scripts/download_model.py
```
If you want to try StyleGAN-Human and the Landscapes HQ (LHQ) dataset, please download weights from these links: [StyleGAN-Human](https://drive.google.com/file/d/1dlFEHbu-WzQWJl7nBBZYcTyo000H9hVm/view?usp=sharing), [LHQ](https://drive.google.com/file/d/16twEf0T9QINAEoMsWefoWiyhcTd-aiWc/view?usp=sharing), and put them under `./checkpoints`.

Feel free to try other pretrained StyleGAN.

## Run DragGAN GUI

To start the DragGAN GUI, simply run:
```sh
sh scripts/gui.sh
```
If you are using windows, you can run:
```
.\scripts\gui.bat
```

This GUI supports editing GAN-generated images. To edit a real image, you need to first perform GAN inversion using tools like [PTI](https://github.com/danielroich/PTI). Then load the new latent code and model weights to the GUI.

You can run DragGAN Gradio demo as well, this is universal for both windows and linux:
```sh
python visualizer_drag_gradio.py
```

## Acknowledgement

This code is developed based on [StyleGAN3](https://github.com/NVlabs/stylegan3). Part of the code is borrowed from [StyleGAN-Human](https://github.com/stylegan-human/StyleGAN-Human).

(cheers to the community as well)
## License

The code related to the DragGAN algorithm is licensed under [CC-BY-NC](https://creativecommons.org/licenses/by-nc/4.0/).
However, most of this project are available under a separate license terms: all codes used or modified from [StyleGAN3](https://github.com/NVlabs/stylegan3) is under the [Nvidia Source Code License](https://github.com/NVlabs/stylegan3/blob/main/LICENSE.txt).

Any form of use and derivative of this code must preserve the watermarking functionality showing "AI Generated".

## BibTeX

```bibtex
@inproceedings{pan2023draggan,
    title={Drag Your GAN: Interactive Point-based Manipulation on the Generative Image Manifold},
    author={Pan, Xingang and Tewari, Ayush, and Leimk{\"u}hler, Thomas and Liu, Lingjie and Meka, Abhimitra and Theobalt, Christian},
    booktitle = {ACM SIGGRAPH 2023 Conference Proceedings},
    year={2023}
}
```
