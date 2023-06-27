@echo off
python visualizer_drag.py ^
    checkpoints/stylegan2_lions_512_pytorch.pkl ^
    checkpoints/stylegan2-ffhq-512x512.pkl ^
    checkpoints/stylegan2-afhqcat-512x512.pkl ^
    checkpoints/stylegan2-car-config-f.pkl ^
    checkpoints/stylegan2_dogs_1024_pytorch.pkl ^
    checkpoints/stylegan2_horses_256_pytorch.pkl ^
    checkpoints/stylegan2-cat-config-f.pkl ^
    checkpoints/stylegan2_elephants_512_pytorch.pkl ^
    checkpoints/stylegan_human_v2_512.pkl ^
    checkpoints/stylegan2-lhq-256x256.pkl
