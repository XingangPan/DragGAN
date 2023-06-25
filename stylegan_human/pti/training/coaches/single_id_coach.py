# Copyright (c) SenseTime Research. All rights reserved.

import os
import torch
from tqdm import tqdm
from pti.pti_configs import paths_config, hyperparameters, global_config
from pti.training.coaches.base_coach import BaseCoach
from utils.log_utils import log_images_from_w
from torchvision.utils import save_image

class SingleIDCoach(BaseCoach):

    def __init__(self, data_loader, use_wandb):
        super().__init__(data_loader, use_wandb)

    def train(self):

        w_path_dir = f'{paths_config.embedding_base_dir}/{paths_config.input_data_id}'
        os.makedirs(w_path_dir, exist_ok=True)
        os.makedirs(f'{w_path_dir}/{paths_config.pti_results_keyword}', exist_ok=True)

        use_ball_holder = True

        for fname, image in tqdm(self.data_loader):
            image_name = fname[0]

            self.restart_training()

            if self.image_counter >= hyperparameters.max_images_to_invert:
                break

            embedding_dir = f'{w_path_dir}/{paths_config.pti_results_keyword}/{image_name}'
            os.makedirs(embedding_dir, exist_ok=True)

            w_pivot = None

            if hyperparameters.use_last_w_pivots:
                w_pivot = self.load_inversions(w_path_dir, image_name)
# Copyright (c) SenseTime Research. All rights reserved.

            elif not hyperparameters.use_last_w_pivots or w_pivot is None:
                w_pivot = self.calc_inversions(image, image_name)

            # w_pivot = w_pivot.detach().clone().to(global_config.device)
            w_pivot = w_pivot.to(global_config.device)

            torch.save(w_pivot, f'{embedding_dir}/0.pt')
            log_images_counter = 0
            real_images_batch = image.to(global_config.device)

            for i in range(hyperparameters.max_pti_steps):

                generated_images = self.forward(w_pivot)
                loss, l2_loss_val, loss_lpips = self.calc_loss(generated_images, real_images_batch, image_name,
                                                               self.G, use_ball_holder, w_pivot)
                if i == 0:
                    tmp1 = torch.clone(generated_images)
                if i % 10 == 0:
                    print("pti loss: ", i, loss.data, loss_lpips.data)
                self.optimizer.zero_grad()

                if loss_lpips <= hyperparameters.LPIPS_value_threshold:
                    break

                loss.backward()
                self.optimizer.step()

                use_ball_holder = global_config.training_step % hyperparameters.locality_regularization_interval == 0

                if self.use_wandb and log_images_counter % global_config.image_rec_result_log_snapshot == 0:
                    log_images_from_w([w_pivot], self.G, [image_name])

                global_config.training_step += 1
                log_images_counter += 1

            # save output image
            tmp = torch.cat([real_images_batch, tmp1, generated_images], axis= 3)
            save_image(tmp, f"{paths_config.experiments_output_dir}/{image_name}.png", normalize=True)


            self.image_counter += 1

            # torch.save(self.G,
            #            f'{paths_config.checkpoints_dir}/model_{image_name}.pt') #'.pt'
            snapshot_data = dict()
            snapshot_data['G_ema'] = self.G
            import pickle
            with open(f'{paths_config.checkpoints_dir}/model_{image_name}.pkl', 'wb') as f:
                    pickle.dump(snapshot_data, f)
