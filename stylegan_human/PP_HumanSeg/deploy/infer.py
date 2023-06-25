# Copyright (c) SenseTime Research. All rights reserved.


# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import codecs
import os
import time

import yaml
import numpy as np
import cv2
import paddle
import paddleseg.transforms as T
from paddle.inference import create_predictor, PrecisionType
from paddle.inference import Config as PredictConfig
from paddleseg.core.infer import reverse_transform
from paddleseg.cvlibs import manager
from paddleseg.utils import TimeAverager

from ..scripts.optic_flow_process import optic_flow_process


class DeployConfig:
    def __init__(self, path):
        with codecs.open(path, 'r', 'utf-8') as file:
            self.dic = yaml.load(file, Loader=yaml.FullLoader)

        self._transforms = self._load_transforms(self.dic['Deploy'][
            'transforms'])
        self._dir = os.path.dirname(path)

    @property
    def transforms(self):
        return self._transforms

    @property
    def model(self):
        return os.path.join(self._dir, self.dic['Deploy']['model'])

    @property
    def params(self):
        return os.path.join(self._dir, self.dic['Deploy']['params'])

    def _load_transforms(self, t_list):
        com = manager.TRANSFORMS
        transforms = []
        for t in t_list:
            ctype = t.pop('type')
            transforms.append(com[ctype](**t))

        return transforms


class Predictor:
    def __init__(self, args):
        self.cfg = DeployConfig(args.cfg)
        self.args = args
        self.compose = T.Compose(self.cfg.transforms)
        resize_h, resize_w = args.input_shape

        self.disflow = cv2.DISOpticalFlow_create(
            cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST)
        self.prev_gray = np.zeros((resize_h, resize_w), np.uint8)
        self.prev_cfd = np.zeros((resize_h, resize_w), np.float32)
        self.is_init = True

        pred_cfg = PredictConfig(self.cfg.model, self.cfg.params)
        pred_cfg.disable_glog_info()
        if self.args.use_gpu:
            pred_cfg.enable_use_gpu(100, 0)

        self.predictor = create_predictor(pred_cfg)
        if self.args.test_speed:
            self.cost_averager = TimeAverager()

    def preprocess(self, img):
        ori_shapes = []
        processed_imgs = []
        processed_img = self.compose(img)[0]
        processed_imgs.append(processed_img)
        ori_shapes.append(img.shape)
        return processed_imgs, ori_shapes

    def run(self, img, bg):
        input_names = self.predictor.get_input_names()
        input_handle = self.predictor.get_input_handle(input_names[0])
        processed_imgs, ori_shapes = self.preprocess(img)
        data = np.array(processed_imgs)
        input_handle.reshape(data.shape)
        input_handle.copy_from_cpu(data)
        if self.args.test_speed:
            start = time.time()

        self.predictor.run()

        if self.args.test_speed:
            self.cost_averager.record(time.time() - start)
        output_names = self.predictor.get_output_names()
        output_handle = self.predictor.get_output_handle(output_names[0])
        output = output_handle.copy_to_cpu()
        return self.postprocess(output, img, ori_shapes[0], bg)


    def postprocess(self, pred, img, ori_shape, bg):
        if not os.path.exists(self.args.save_dir):
            os.makedirs(self.args.save_dir)
        resize_w = pred.shape[-1]
        resize_h = pred.shape[-2]
        if self.args.soft_predict:
            if self.args.use_optic_flow:
                score_map = pred[:, 1, :, :].squeeze(0)
                score_map = 255 * score_map
                cur_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                cur_gray = cv2.resize(cur_gray, (resize_w, resize_h))
                optflow_map = optic_flow_process(cur_gray, score_map, self.prev_gray, self.prev_cfd, \
                        self.disflow, self.is_init)
                self.prev_gray = cur_gray.copy()
                self.prev_cfd = optflow_map.copy()
                self.is_init = False

                score_map = np.repeat(optflow_map[:, :, np.newaxis], 3, axis=2)
                score_map = np.transpose(score_map, [2, 0, 1])[np.newaxis, ...]
                score_map = reverse_transform(
                    paddle.to_tensor(score_map),
                    ori_shape,
                    self.cfg.transforms,
                    mode='bilinear')
                alpha = np.transpose(score_map.numpy().squeeze(0),
                                     [1, 2, 0]) / 255
            else:
                score_map = pred[:, 1, :, :]
                score_map = score_map[np.newaxis, ...]
                score_map = reverse_transform(
                    paddle.to_tensor(score_map),
                    ori_shape,
                    self.cfg.transforms,
                    mode='bilinear')
                alpha = np.transpose(score_map.numpy().squeeze(0), [1, 2, 0])

        else:
            if pred.ndim == 3:
                pred = pred[:, np.newaxis, ...]
            result = reverse_transform(
                paddle.to_tensor(
                    pred, dtype='float32'),
                ori_shape,
                self.cfg.transforms,
                mode='bilinear')

            result = np.array(result)
            if self.args.add_argmax:
                result = np.argmax(result, axis=1)
            else:
                result = result.squeeze(1)
            alpha = np.transpose(result, [1, 2, 0])

        # background replace
        h, w, _ = img.shape
        if bg is None:
            bg = np.ones_like(img)*255
        else:
            bg = cv2.resize(bg, (w, h))
            if bg.ndim == 2:
                bg = bg[..., np.newaxis]

        comb = (alpha * img + (1 - alpha) * bg).astype(np.uint8)
        return comb, alpha, bg, img
