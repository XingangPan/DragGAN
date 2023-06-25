# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import numpy as np
import imgui
import dnnlib
import torch
from gui_utils import imgui_utils

#----------------------------------------------------------------------------

class LatentWidget:
    def __init__(self, viz):
        self.viz        = viz
        self.seed       = 0
        self.w_plus     = True
        self.reg        = 0
        self.lr         = 0.001
        self.w_path     = ''
        self.w_load     = None
        self.defer_frames   = 0
        self.disabled_time  = 0

    @imgui_utils.scoped_by_object_id
    def __call__(self, show=True):
        viz = self.viz
        if show:
            with imgui_utils.grayed_out(self.disabled_time != 0):
                imgui.text('Latent')
                imgui.same_line(viz.label_w)
                with imgui_utils.item_width(viz.font_size * 8.75):
                    changed, seed = imgui.input_int('Seed', self.seed)
                    if changed:
                        self.seed = seed
                        # reset latent code
                        self.w_load = None

                # load latent code
                imgui.text(' ')
                imgui.same_line(viz.label_w)
                _changed, self.w_path = imgui_utils.input_text('##path', self.w_path, 1024,
                    flags=(imgui.INPUT_TEXT_AUTO_SELECT_ALL | imgui.INPUT_TEXT_ENTER_RETURNS_TRUE),
                    width=(-1),
                    help_text='Path to latent code')
                if imgui.is_item_hovered() and not imgui.is_item_active() and self.w_path != '':
                    imgui.set_tooltip(self.w_path)

                imgui.text(' ')
                imgui.same_line(viz.label_w)
                if imgui_utils.button('Load latent', width=viz.button_w, enabled=(self.disabled_time == 0 and 'image' in viz.result)):
                    assert os.path.isfile(self.w_path), f"{self.w_path} does not exist!"
                    self.w_load = torch.load(self.w_path)
                    self.defer_frames = 2
                    self.disabled_time = 0.5
                
                imgui.text(' ')
                imgui.same_line(viz.label_w)
                with imgui_utils.item_width(viz.button_w):
                    changed, lr = imgui.input_float('Step Size', self.lr)
                    if changed:
                        self.lr = lr

                # imgui.text(' ')
                # imgui.same_line(viz.label_w)
                # with imgui_utils.item_width(viz.button_w):
                #     changed, reg = imgui.input_float('Regularize', self.reg)
                #     if changed:
                #         self.reg = reg

                imgui.text(' ')
                imgui.same_line(viz.label_w)
                reset_w = imgui_utils.button('Reset', width=viz.button_w, enabled='image' in viz.result)
                imgui.same_line()
                _clicked, w = imgui.checkbox('w', not self.w_plus)
                if w:
                    self.w_plus = False
                imgui.same_line()
                _clicked, self.w_plus = imgui.checkbox('w+', self.w_plus)

        self.disabled_time = max(self.disabled_time - viz.frame_delta, 0)
        if self.defer_frames > 0:
            self.defer_frames -= 1
        viz.args.w0_seed = self.seed
        viz.args.w_load = self.w_load
        viz.args.reg = self.reg
        viz.args.w_plus = self.w_plus
        viz.args.reset_w = reset_w
        viz.args.lr = lr

#----------------------------------------------------------------------------
