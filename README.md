# Drag Your GAN: Interactive Point-based Manipulation on the Generative Image Manifold

<p align="center">
    <img src="DragGAN.gif", width="700">
</p>

**Figure:** *Drag your GAN.*

> **Drag Your GAN: Interactive Point-based Manipulation on the Generative Image Manifold** <br>
> Xingang Pan, Ayush Tewari, Thomas Leimkühler, Lingjie Liu, Abhimitra Meka, Christian Theobalt<br>
> *SIGGRAPH 2023 Conference Proceedings*

## Requirements

Please follow the requirements of [https://github.com/NVlabs/stylegan3](https://github.com/NVlabs/stylegan3).

## Download pre-trained StyleGAN2 weights

To download pre-trained weights, simply run:
```sh
sh scripts/download_model.sh
```
If you want to try StyleGAN-Human and the Landscapes HQ (LHQ) dataset, please download weights from these links: [StyleGAN-Human](https://drive.google.com/file/d/1dlFEHbu-WzQWJl7nBBZYcTyo000H9hVm/view?usp=sharing), [LHQ](https://drive.google.com/file/d/16twEf0T9QINAEoMsWefoWiyhcTd-aiWc/view?usp=sharing), and put them under `./checkpoints`.

Feel free to try other pretrained StyleGAN.

## Run DragGAN GUI

To start the DragGAN GUI, simply run:
```sh
sh scripts/gui.sh
```

This GUI supports editing GAN-generated images. To edit a real image, you need to first perform GAN inversion using tools like [PTI](https://github.com/danielroich/PTI). Then load the new latent code and model weights to the GUI.

You can run DragGAN Gradio demo as well:
```sh
python visualizer_drag_gradio.py
```

## Acknowledgement

This code is developed based on [StyleGAN3](https://github.com/NVlabs/stylegan3). Part of the code is borrowed from [StyleGAN-Human](https://github.com/stylegan-human/StyleGAN-Human).

## License

The code related to the DragGAN algorithm is licensed under [CC-BY-NC](https://creativecommons.org/licenses/by-nc/4.0/).
However, most of this project are available under a separate license terms: all codes used or modified from [StyleGAN3](https://github.com/NVlabs/stylegan3) is under the [Nvidia Source Code License](https://github.com/NVlabs/stylegan3/blob/main/LICENSE.txt).

Any form of use and derivative of this code must preserve the watermarking functionality.

## BibTeX

```bibtex
@inproceedings{pan2023draggan,
    title={Drag Your GAN: Interactive Point-based Manipulation on the Generative Image Manifold},
    author={Pan, Xingang and Tewari, Ayush, and Leimk{\"u}hler, Thomas and Liu, Lingjie and Meka, Abhimitra and Theobalt, Christian},
    booktitle = {ACM SIGGRAPH 2023 Conference Proceedings},
    year={2023}
}
```
