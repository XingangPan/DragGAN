# Copyright (c) SenseTime Research. All rights reserved.


import torch
import torch.nn.functional as F
from tqdm import tqdm
from lpips import LPIPS
import numpy as np
from torch_utils.models import Generator as bodyGAN
from torch_utils.models_face import Generator as FaceGAN
import dlib
from utils.face_alignment import align_face_for_insetgan
from utils.util import visual,tensor_to_numpy, numpy_to_tensor
import legacy
import os
import click


class InsetGAN(torch.nn.Module):
    def __init__(self, stylebody_ckpt, styleface_ckpt):
        super().__init__()
        
        ## convert pkl to pth
        if not os.path.exists(stylebody_ckpt.replace('.pkl','.pth')):
            legacy.convert(stylebody_ckpt, stylebody_ckpt.replace('.pkl','.pth'))
        stylebody_ckpt = stylebody_ckpt.replace('.pkl','.pth') 
        
        if not os.path.exists(styleface_ckpt.replace('.pkl','.pth')):
            legacy.convert(styleface_ckpt, styleface_ckpt.replace('.pkl','.pth'))
        styleface_ckpt = styleface_ckpt.replace('.pkl','.pth') 
    
        # dual generator
        config = {"latent" : 512, "n_mlp" : 8, "channel_multiplier": 2}
        self.body_generator = bodyGAN(
                size = 1024,
                style_dim=config["latent"],
                n_mlp=config["n_mlp"],
                channel_multiplier=config["channel_multiplier"]
            )
        self.body_generator.load_state_dict(torch.load(stylebody_ckpt)['g_ema'])
        self.body_generator.eval().requires_grad_(False).cuda()

        self.face_generator = FaceGAN(
                size = 1024,
                style_dim=config["latent"],
                n_mlp=config["n_mlp"],
                channel_multiplier=config["channel_multiplier"]
            )
        self.face_generator.load_state_dict(torch.load(styleface_ckpt)['g_ema'])
        self.face_generator.eval().requires_grad_(False).cuda()
        # crop function
        self.dlib_predictor = dlib.shape_predictor('./pretrained_models/shape_predictor_68_face_landmarks.dat')
        self.dlib_cnn_face_detector = dlib.cnn_face_detection_model_v1("pretrained_models/mmod_human_face_detector.dat")

        # criterion
        self.lpips_loss = LPIPS(net='alex').cuda().eval()
        self.l1_loss = torch.nn.L1Loss(reduction='mean')
        
    def loss_coarse(self, A_face, B, p1=500, p2=0.05):
        A_face = F.interpolate(A_face, size=(64, 64), mode='area')
        B = F.interpolate(B, size=(64, 64), mode='area')
        loss_l1 = p1 * self.l1_loss(A_face, B)
        loss_lpips = p2 * self.lpips_loss(A_face, B)
        return loss_l1 + loss_lpips

    @staticmethod
    def get_border_mask(A, x, spec):
        mask = torch.zeros_like(A)
        mask[:, :, :x, ] = 1
        mask[:, :, -x:, ] = 1
        mask[:, :, :, :x ] = 1
        mask[:, :, :, -x:] = 1
        return mask
    
    @staticmethod
    def get_body_mask(A, crop, padding=4):
        mask = torch.ones_like(A)
        mask[:, :, crop[1]-padding:crop[3]+padding, crop[0]-padding:crop[2]+padding] = 0
        return mask

    def loss_border(self, A_face, B, p1=10000, p2=2, spec=None):
        mask = self.get_border_mask(A_face, 8, spec)
        loss_l1 = p1 * self.l1_loss(A_face*mask, B*mask)
        loss_lpips = p2 * self.lpips_loss(A_face*mask, B*mask)
        return loss_l1 + loss_lpips

    def loss_body(self, A, B, crop, p1=9000, p2=0.1):
        padding = int((crop[3] - crop[1]) / 20)
        mask = self.get_body_mask(A, crop, padding)
        loss_l1 = p1 * self.l1_loss(A*mask, B*mask)
        loss_lpips = p2 * self.lpips_loss(A*mask, B*mask)
        return loss_l1+loss_lpips

    def loss_face(self, A, B, crop, p1=5000, p2=1.75):
        mask = 1 - self.get_body_mask(A, crop)
        loss_l1 = p1 * self.l1_loss(A*mask, B*mask)
        loss_lpips = p2 * self.lpips_loss(A*mask, B*mask)
        return loss_l1+loss_lpips
    
    def loss_reg(self, w, w_mean, p1, w_plus_delta=None, p2=None):
        return p1 * torch.mean(((w - w_mean) ** 2)) + p2 * torch.mean(w_plus_delta ** 2)
   
    # FFHQ type 
    def detect_face_dlib(self, img):
        # tensor to numpy array rgb uint8
        img = tensor_to_numpy(img)
        aligned_image, crop, rect = align_face_for_insetgan(img=img, 
                                               detector=self.dlib_cnn_face_detector, 
                                               predictor=self.dlib_predictor, 
                                               output_size=256)

        aligned_image = np.array(aligned_image)
        aligned_image = numpy_to_tensor(aligned_image)
        return aligned_image, crop, rect
  
    # joint optimization
    def dual_optimizer(self, 
                       face_w,
                       body_w,
                       joint_steps=500,
                       face_initial_learning_rate=0.02,
                       body_initial_learning_rate=0.05,
                       lr_rampdown_length=0.25,
                       lr_rampup_length=0.05,
                       seed=None,
                       output_path=None,
                       video=0): 
        '''
        Given a face_w, optimize a body_w with suitable body pose & shape for face_w
        '''
        def visual_(path, synth_body, synth_face, body_crop, step, both=False, init_body_with_face=None):
            tmp = synth_body.clone().detach()
            tmp[:, :, body_crop[1]:body_crop[3], body_crop[0]:body_crop[2]] = synth_face
            if both:
                tmp = torch.cat([synth_body, tmp], dim=3)
            save_path = os.path.join(path, f"{step:04d}.jpg")
            visual(tmp, save_path)
            
        def forward(face_w_opt, 
                    body_w_opt, 
                    face_w_delta,
                    body_w_delta, 
                    body_crop, 
                    update_crop=False
                    ):
            if face_w_opt.shape[1] != 18:
                face_ws = (face_w_opt).repeat([1, 18, 1])
            else:
                face_ws = face_w_opt.clone()
            face_ws = face_ws + face_w_delta
            synth_face, _ = self.face_generator([face_ws], input_is_latent=True, randomize_noise=False)
            
            body_ws = (body_w_opt).repeat([1, 18, 1])
            body_ws = body_ws + body_w_delta
            synth_body, _ = self.body_generator([body_ws], input_is_latent=True, randomize_noise=False)
            
            if update_crop:
                old_r = (body_crop[3]-body_crop[1]) // 2, (body_crop[2]-body_crop[0]) // 2
                _, body_crop, _ = self.detect_face_dlib(synth_body)
                center = (body_crop[1] + body_crop[3]) // 2, (body_crop[0] + body_crop[2]) // 2
                body_crop = (center[1] - old_r[1], center[0] - old_r[0], center[1] + old_r[1], center[0] + old_r[0])
            
            synth_body_face = synth_body[:, :, body_crop[1]:body_crop[3], body_crop[0]:body_crop[2]]
            
            if synth_face.shape[2] > body_crop[3]-body_crop[1]:
                synth_face_resize = F.interpolate(synth_face, size=(body_crop[3]-body_crop[1], body_crop[2]-body_crop[0]), mode='area')
        
            return synth_body, synth_body_face, synth_face, synth_face_resize, body_crop
                
        def update_lr(init_lr, step, num_steps, lr_rampdown_length, lr_rampup_length):
            t = step / num_steps
            lr_ramp = min(1.0, (1.0 - t) / lr_rampdown_length)
            lr_ramp = 0.5 - 0.5 * np.cos(lr_ramp * np.pi)
            lr_ramp = lr_ramp * min(1.0, t / lr_rampup_length)
            lr = init_lr * lr_ramp
            return lr
        
        # update output_path
        output_path = os.path.join(output_path, seed)
        os.makedirs(output_path, exist_ok=True)
        
        # define optimized params
        body_w_mean = self.body_generator.mean_latent(10000).detach()
        face_w_opt = face_w.clone().detach().requires_grad_(True)
        body_w_opt = body_w.clone().detach().requires_grad_(True)
        face_w_delta = torch.zeros_like(face_w.repeat([1, 18, 1])).requires_grad_(True)
        body_w_delta = torch.zeros_like(body_w.repeat([1, 18, 1])).requires_grad_(True)
        # generate ref face & body
        ref_body, _ = self.body_generator([body_w.repeat([1, 18, 1])], input_is_latent=True, randomize_noise=False)
        # for inversion
        ref_face, _ = self.face_generator([face_w.repeat([1, 18, 1])], input_is_latent=True, randomize_noise=False)
        # get initilized crop
        _, body_crop, _ = self.detect_face_dlib(ref_body)
        _, _, face_crop = self.detect_face_dlib(ref_face) # NOTE: this is face rect only. no FFHQ type.
        # create optimizer
        face_optimizer = torch.optim.Adam([face_w_opt, face_w_delta], betas=(0.9, 0.999), lr=face_initial_learning_rate)
        body_optimizer = torch.optim.Adam([body_w_opt, body_w_delta], betas=(0.9, 0.999), lr=body_initial_learning_rate)
        
        global_step = 0
        # Stage1: remove background of face image
        face_steps = 25
        pbar = tqdm(range(face_steps))
        for step in pbar:
            face_lr = update_lr(face_initial_learning_rate / 2, step, face_steps, lr_rampdown_length, lr_rampup_length)
            for param_group in face_optimizer.param_groups:
                param_group['lr'] =face_lr
            synth_body, synth_body_face, synth_face_raw, synth_face, body_crop = forward(face_w_opt, 
                                                                                    body_w_opt, 
                                                                                    face_w_delta,
                                                                                    body_w_delta, 
                                                                                    body_crop)
            loss_face = self.loss_face(synth_face_raw, ref_face, face_crop, 5000, 1.75)
            loss_coarse = self.loss_coarse(synth_face, synth_body_face, 50, 0.05)
            loss_border = self.loss_border(synth_face, synth_body_face, 1000, 0.1)
            loss = loss_coarse + loss_border + loss_face
            face_optimizer.zero_grad()
            loss.backward()
            face_optimizer.step()
            # visualization
            if video:
                visual_(output_path, synth_body, synth_face, body_crop, global_step)
            pbar.set_description(
                (                
                    f"face: {step:.4f}, lr: {face_lr}, loss: {loss.item():.2f}, loss_coarse: {loss_coarse.item():.2f};"
                    f"loss_border: {loss_border.item():.2f}, loss_face: {loss_face.item():.2f};"
                )
            )
            global_step += 1
            
        # Stage2: find a suitable body
        body_steps = 150
        pbar = tqdm(range(body_steps))
        for step in pbar:
            body_lr = update_lr(body_initial_learning_rate, step, body_steps, lr_rampdown_length, lr_rampup_length)
            update_crop = True if (step % 50 == 0) else False
            # update_crop = False
            for param_group in body_optimizer.param_groups:
                param_group['lr'] =body_lr
            synth_body, synth_body_face, synth_face_raw, synth_face, body_crop = forward(face_w_opt, 
                                                                                    body_w_opt, 
                                                                                    face_w_delta, 
                                                                                    body_w_delta, 
                                                                                    body_crop,
                                                                                    update_crop=update_crop)
            loss_coarse = self.loss_coarse(synth_face, synth_body_face, 500, 0.05)
            loss_border = self.loss_border(synth_face, synth_body_face, 2500, 0)
            loss_body = self.loss_body(synth_body, ref_body, body_crop, 9000, 0.1)
            loss_reg = self.loss_reg(body_w_opt, body_w_mean, 15000, body_w_delta, 0)
            loss = loss_coarse + loss_border + loss_body + loss_reg
            body_optimizer.zero_grad()
            loss.backward()
            body_optimizer.step()     
            
            # visualization
            if video:
                visual_(output_path, synth_body, synth_face, body_crop, global_step)
            pbar.set_description(
                (
                    f"body: {step:.4f}, lr: {body_lr}, loss: {loss.item():.2f}, loss_coarse: {loss_coarse.item():.2f};"
                    f"loss_border: {loss_border.item():.2f}, loss_body: {loss_body.item():.2f}, loss_reg: {loss_reg:.2f}"
                )
            )    
            global_step += 1
        
        # Stage3: joint optimization
        interval = 50
        joint_face_steps = joint_steps // 2
        joint_body_steps = joint_steps // 2
        face_step = 0
        body_step = 0
        pbar = tqdm(range(joint_steps))
        flag = -1
        for step in pbar:
            if step % interval == 0: flag += 1
            text_flag = 'optimize_face' if flag % 2 == 0 else 'optimize_body'
            synth_body, synth_body_face, synth_face_raw, synth_face, body_crop = forward(face_w_opt, 
                                                                                    body_w_opt, 
                                                                                    face_w_delta, 
                                                                                    body_w_delta, 
                                                                                    body_crop)
            if text_flag == 'optimize_face':
                face_lr = update_lr(face_initial_learning_rate, face_step, joint_face_steps, lr_rampdown_length, lr_rampup_length)
                for param_group in face_optimizer.param_groups:
                    param_group['lr'] =face_lr
                loss_face = self.loss_face(synth_face_raw, ref_face, face_crop, 5000, 1.75)
                loss_coarse = self.loss_coarse(synth_face, synth_body_face, 500, 0.05)
                loss_border = self.loss_border(synth_face, synth_body_face, 25000, 0)
                loss = loss_coarse + loss_border + loss_face
                face_optimizer.zero_grad()
                loss.backward()
                face_optimizer.step()
                pbar.set_description(
                    (                
                        f"face: {step}, lr: {face_lr:.4f}, loss: {loss.item():.2f}, loss_coarse: {loss_coarse.item():.2f};"
                        f"loss_border: {loss_border.item():.2f}, loss_face: {loss_face.item():.2f};"
                    )
                )
                face_step += 1
            else:
                body_lr = update_lr(body_initial_learning_rate, body_step, joint_body_steps, lr_rampdown_length, lr_rampup_length)
                for param_group in body_optimizer.param_groups:
                    param_group['lr'] =body_lr
                loss_coarse = self.loss_coarse(synth_face, synth_body_face, 500, 0.05)
                loss_border = self.loss_border(synth_face, synth_body_face, 2500, 0)
                loss_body = self.loss_body(synth_body, ref_body, body_crop, 9000, 0.1)
                loss_reg = self.loss_reg(body_w_opt, body_w_mean, 25000, body_w_delta, 0)
                loss = loss_coarse + loss_border + loss_body + loss_reg
                body_optimizer.zero_grad()
                loss.backward()
                body_optimizer.step()
                pbar.set_description(
                    (
                        f"body: {step}, lr: {body_lr:.4f}, loss: {loss.item():.2f}, loss_coarse: {loss_coarse.item():.2f};"
                        f"loss_border: {loss_border.item():.2f}, loss_body: {loss_body.item():.2f}, loss_reg: {loss_reg:.2f}"
                    )
                )
                body_step += 1
            if video:
                visual_(output_path, synth_body, synth_face, body_crop, global_step)
            global_step += 1
        return face_w_opt.repeat([1, 18, 1])+face_w_delta, body_w_opt.repeat([1, 18, 1])+body_w_delta, body_crop




"""
Jointly combine and optimize generated faces and bodies .
Examples:

\b
# Combine the generate human full-body image from the provided StyleGAN-Human pre-trained model
# and the generated face image from FFHQ model, optimize both latent codes to produce the coherent face-body image
python insetgan.py --body_network=pretrained_models/stylegan_human_v2_1024.pkl --face_network=pretrained_models/ffhq.pkl \\
    --body_seed=82 --face_seed=43  --trunc=0.6 --outdir=outputs/insetgan/ --video 1 
"""

@click.command()
@click.pass_context
@click.option('--face_network', default="./pretrained_models/ffhq.pkl", help='Network pickle filename', required=True)
@click.option('--body_network', default='./pretrained_models/stylegan2_1024.pkl', help='Network pickle filename', required=True)
@click.option('--face_seed', type=int, default=82, help='selected random seed')
@click.option('--body_seed', type=int, default=43, help='selected random seed')
@click.option('--joint_steps', type=int, default=500, help='num steps for joint optimization')
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=0.6, show_default=True)
@click.option('--outdir', help='Where to save the output images', default= "outputs/insetgan/" , type=str, required=True, metavar='DIR')
@click.option('--video', help="set to 1 if want to save video", type=int, default=0)
def main(
        ctx: click.Context,
        face_network: str,
        body_network: str,
        face_seed: int,
        body_seed: int,
        joint_steps: int,
        truncation_psi: float,
        outdir: str,
        video: int):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    insgan = InsetGAN(body_network, face_network)
    os.makedirs(outdir, exist_ok=True)
    face_z = np.random.RandomState(face_seed).randn(1, 512).astype(np.float32)
    face_mean = insgan.face_generator.mean_latent(3000)
    face_w = insgan.face_generator.get_latent(torch.from_numpy(face_z).to(device))  # [N, L, C]
    face_w = truncation_psi * face_w + (1-truncation_psi) * face_mean
    face_img, _ = insgan.face_generator([face_w], input_is_latent=True)

    body_z = np.random.RandomState(body_seed).randn(1, 512).astype(np.float32)
    body_mean = insgan.body_generator.mean_latent(3000)
    body_w = insgan.body_generator.get_latent(torch.from_numpy(body_z).to(device))  # [N, L, C]
    body_w = truncation_psi * body_w + (1-truncation_psi) * body_mean
    body_img, _ = insgan.body_generator([body_w], input_is_latent=True)

    _, body_crop, _ = insgan.detect_face_dlib(body_img)
    face_img = F.interpolate(face_img, size=(body_crop[3]-body_crop[1], body_crop[2]-body_crop[0]), mode='area')
    cp_body = body_img.clone()
    cp_body[:, :, body_crop[1]:body_crop[3], body_crop[0]:body_crop[2]] = face_img
    
    optim_face_w, optim_body_w, crop = insgan.dual_optimizer(
        face_w, 
        body_w,
        joint_steps=joint_steps,
        seed=f'{face_seed:04d}_{body_seed:04d}',
        output_path=outdir,
        video=video
    )
    
    if video:
        ffmpeg_cmd = f"ffmpeg -hide_banner -loglevel error -i ./{outdir}/{face_seed:04d}_{body_seed:04d}/%04d.jpg -c:v libx264 -vf fps=30 -pix_fmt yuv420p ./{outdir}/{face_seed:04d}_{body_seed:04d}.mp4"
        os.system(ffmpeg_cmd)
    new_face_img, _ = insgan.face_generator([optim_face_w], input_is_latent=True)
    new_shape = crop[3] - crop[1], crop[2] - crop[0]
    new_face_img_crop = F.interpolate(new_face_img, size=new_shape, mode='area')
    seamless_body, _ = insgan.body_generator([optim_body_w], input_is_latent=True)
    seamless_body[:, :, crop[1]:crop[3], crop[0]:crop[2]] = new_face_img_crop
    temp = torch.cat([cp_body, seamless_body], dim=3)
    visual(temp, f"{outdir}/{face_seed:04d}_{body_seed:04d}.png")

if __name__ == "__main__":
    main()