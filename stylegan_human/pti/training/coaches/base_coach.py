import abc
import os
import pickle
from argparse import Namespace
import wandb
import os.path
from .localitly_regulizer import Space_Regulizer, l2_loss
import torch
from torchvision import transforms
from lpips import LPIPS
from pti.training.projectors import w_projector
from pti.pti_configs import global_config, paths_config, hyperparameters
from pti.pti_models.e4e.psp import pSp
from utils.log_utils import log_image_from_w
from utils.models_utils import toogle_grad, load_old_G


class BaseCoach:
    def __init__(self, data_loader, use_wandb):

        self.use_wandb = use_wandb
        self.data_loader = data_loader
        self.w_pivots = {}
        self.image_counter = 0

        if hyperparameters.first_inv_type == 'w+':
            self.initilize_e4e()

        self.e4e_image_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 128)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

        # Initialize loss
        self.lpips_loss = LPIPS(net=hyperparameters.lpips_type).to(global_config.device).eval()

        self.restart_training()

        # Initialize checkpoint dir
        self.checkpoint_dir = paths_config.checkpoints_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def restart_training(self):

        # Initialize networks
        self.G = load_old_G()
        toogle_grad(self.G, True)

        self.original_G = load_old_G()

        self.space_regulizer = Space_Regulizer(self.original_G, self.lpips_loss)
        self.optimizer = self.configure_optimizers()

    def get_inversion(self, w_path_dir, image_name, image):
        embedding_dir = f'{w_path_dir}/{paths_config.pti_results_keyword}/{image_name}'
        os.makedirs(embedding_dir, exist_ok=True)

        w_pivot = None

        if hyperparameters.use_last_w_pivots:
            w_pivot = self.load_inversions(w_path_dir, image_name)

        if not hyperparameters.use_last_w_pivots or w_pivot is None:
            w_pivot = self.calc_inversions(image, image_name)
            torch.save(w_pivot, f'{embedding_dir}/0.pt')

        w_pivot = w_pivot.to(global_config.device)
        return w_pivot

    def load_inversions(self, w_path_dir, image_name):
        if image_name in self.w_pivots:
            return self.w_pivots[image_name]

        if hyperparameters.first_inv_type == 'w+':
            w_potential_path = f'{w_path_dir}/{paths_config.e4e_results_keyword}/{image_name}/0.pt'
        else:
            w_potential_path = f'{w_path_dir}/{paths_config.pti_results_keyword}/{image_name}/0.pt'
        if not os.path.isfile(w_potential_path):
            return None
        w = torch.load(w_potential_path).to(global_config.device)
        self.w_pivots[image_name] = w
        return w

    def calc_inversions(self, image, image_name):
        if hyperparameters.first_inv_type == 'w+':
            w = self.get_e4e_inversion(image)

        else:
            id_image = torch.squeeze((image.to(global_config.device) + 1) / 2) * 255
            w = w_projector.project(self.G, id_image, device=torch.device(global_config.device), w_avg_samples=600,
                                    num_steps=hyperparameters.first_inv_steps, w_name=image_name,
                                    use_wandb=self.use_wandb)

        return w

    @abc.abstractmethod
    def train(self):
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.G.parameters(), lr=hyperparameters.pti_learning_rate)

        return optimizer

    def calc_loss(self, generated_images, real_images, log_name, new_G, use_ball_holder, w_batch):
        loss = 0.0

        if hyperparameters.pt_l2_lambda > 0:
            l2_loss_val = l2_loss(generated_images, real_images)
            if self.use_wandb:
                wandb.log({f'MSE_loss_val_{log_name}': l2_loss_val.detach().cpu()}, step=global_config.training_step)
            loss += l2_loss_val * hyperparameters.pt_l2_lambda
        if hyperparameters.pt_lpips_lambda > 0:
            loss_lpips = self.lpips_loss(generated_images, real_images)
            loss_lpips = torch.squeeze(loss_lpips)
            if self.use_wandb:
                wandb.log({f'LPIPS_loss_val_{log_name}': loss_lpips.detach().cpu()}, step=global_config.training_step)
            loss += loss_lpips * hyperparameters.pt_lpips_lambda

        if use_ball_holder and hyperparameters.use_locality_regularization:
            ball_holder_loss_val = self.space_regulizer.space_regulizer_loss(new_G, w_batch, use_wandb=self.use_wandb)
            loss += ball_holder_loss_val

        return loss, l2_loss_val, loss_lpips

    def forward(self, w):
        generated_images = self.G.synthesis(w, noise_mode='const', force_fp32=True)

        return generated_images

    def initilize_e4e(self):
        ckpt = torch.load(paths_config.e4e, map_location='cpu')
        opts = ckpt['opts']
        opts['batch_size'] = hyperparameters.train_batch_size
        opts['checkpoint_path'] = paths_config.e4e
        opts = Namespace(**opts)
        self.e4e_inversion_net = pSp(opts)
        self.e4e_inversion_net.eval()
        self.e4e_inversion_net = self.e4e_inversion_net.to(global_config.device)
        toogle_grad(self.e4e_inversion_net, False)

    def get_e4e_inversion(self, image):
        image = (image + 1) / 2
        new_image = self.e4e_image_transform(image[0]).to(global_config.device)
        _, w = self.e4e_inversion_net(new_image.unsqueeze(0), randomize_noise=False, return_latents=True, resize=False,
                                      input_code=False)
        if self.use_wandb:
            log_image_from_w(w, self.G, 'First e4e inversion')
        return w
