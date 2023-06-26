@echo off
mkdir checkpoints
cd checkpoints

powershell -Command "(New-Object System.Net.WebClient).DownloadFile('https://storage.googleapis.com/self-distilled-stylegan/lions_512_pytorch.pkl', 'lions_512_pytorch.pkl')"
ren lions_512_pytorch.pkl stylegan2_lions_512_pytorch.pkl

powershell -Command "(New-Object System.Net.WebClient).DownloadFile('https://storage.googleapis.com/self-distilled-stylegan/dogs_1024_pytorch.pkl', 'dogs_1024_pytorch.pkl')"
ren dogs_1024_pytorch.pkl stylegan2_dogs_1024_pytorch.pkl

powershell -Command "(New-Object System.Net.WebClient).DownloadFile('https://storage.googleapis.com/self-distilled-stylegan/horses_256_pytorch.pkl', 'horses_256_pytorch.pkl')"
ren horses_256_pytorch.pkl stylegan2_horses_256_pytorch.pkl

powershell -Command "(New-Object System.Net.WebClient).DownloadFile('https://storage.googleapis.com/self-distilled-stylegan/elephants_512_pytorch.pkl', 'elephants_512_pytorch.pkl')"
ren elephants_512_pytorch.pkl stylegan2_elephants_512_pytorch.pkl

powershell -Command "(New-Object System.Net.WebClient).DownloadFile('https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan2/versions/1/files/stylegan2-ffhq-512x512.pkl', 'stylegan2-ffhq-512x512.pkl')"
powershell -Command "(New-Object System.Net.WebClient).DownloadFile('https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan2/versions/1/files/stylegan2-afhqcat-512x512.pkl', 'stylegan2-afhqcat-512x512.pkl')"
powershell -Command "(New-Object System.Net.WebClient).DownloadFile('http://d36zk2xti64re0.cloudfront.net/stylegan2/networks/stylegan2-car-config-f.pkl', 'stylegan2-car-config-f.pkl')"
powershell -Command "(New-Object System.Net.WebClient).DownloadFile('http://d36zk2xti64re0.cloudfront.net/stylegan2/networks/stylegan2-cat-config-f.pkl', 'stylegan2-cat-config-f.pkl')"

echo "Done"
pause
