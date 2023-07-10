import torch
from inversion import  inverse_image,get_lr

from tqdm import tqdm
from torch.nn import functional as F
from lpips import util
def toogle_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


class PTI:
    def __init__(self,G,l2_lambda = 1,max_pti_step = 400, pti_lr = 3e-4 ):
        self.g_ema = G
        self.l2_lambda = l2_lambda
        self.max_pti_step = max_pti_step
        self.pti_lr = pti_lr
    def cacl_loss(self,percept, generated_image,real_image):

        mse_loss = F.mse_loss(generated_image, real_image)
        p_loss = percept(generated_image, real_image).sum()
        loss = p_loss +self.l2_lambda * mse_loss
        return loss

    def train(self,img):
        inversed_result = inverse_image(self.g_ema,img,self.g_ema.img_resolution)
        w_pivot = inversed_result['latent']
        ws = w_pivot.repeat([1, self.g_ema.mapping.num_ws, 1])
        toogle_grad(self.g_ema,True)
        percept = util.PerceptualLoss(
            model="net-lin", net="vgg", use_gpu='cuda:0'
        )
        optimizer = torch.optim.Adam(self.g_ema.parameters(), lr=self.pti_lr)
        print('start PTI')
        pbar = tqdm(range(self.max_pti_step))
        for i in pbar:
            lr = get_lr(i, self.pti_lr)
            optimizer.param_groups[0]["lr"] = lr

            generated_image,feature = self.g_ema.synthesis(ws,noise_mode='const')
            loss = self.cacl_loss(percept,generated_image,inversed_result['real'])
            pbar.set_description(
                (
                    f"loss: {loss.item():.4f}"
                )
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            generated_image = self.g_ema.synthesis(ws, noise_mode='const')

        return generated_image
