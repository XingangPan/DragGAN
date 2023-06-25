mkdir checkpoints
cd checkpoints

wget https://storage.googleapis.com/self-distilled-stylegan/lions_512_pytorch.pkl
mv lions_512_pytorch.pkl stylegan2_lions_512_pytorch.pkl

wget https://storage.googleapis.com/self-distilled-stylegan/dogs_1024_pytorch.pkl
mv dogs_1024_pytorch.pkl stylegan2_dogs_1024_pytorch.pkl

wget https://storage.googleapis.com/self-distilled-stylegan/horses_256_pytorch.pkl
mv horses_256_pytorch.pkl stylegan2_horses_256_pytorch.pkl

wget https://storage.googleapis.com/self-distilled-stylegan/elephants_512_pytorch.pkl
mv elephants_512_pytorch.pkl stylegan2_elephants_512_pytorch.pkl

wget https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan2/versions/1/files/stylegan2-ffhq-512x512.pkl
wget https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan2/versions/1/files/stylegan2-afhqcat-512x512.pkl
wget http://d36zk2xti64re0.cloudfront.net/stylegan2/networks/stylegan2-car-config-f.pkl
wget http://d36zk2xti64re0.cloudfront.net/stylegan2/networks/stylegan2-cat-config-f.pkl
