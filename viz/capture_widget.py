# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import re
import numpy as np
import imgui
import PIL.Image
from gui_utils import imgui_utils
from . import renderer
import torch
import torchvision

#----------------------------------------------------------------------------

class CaptureWidget:
    def __init__(self, viz):
        self.viz            = viz
        self.path           = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '_screenshots'))
        self.dump_image     = False
        self.dump_gui       = False
        self.defer_frames   = 0
        self.disabled_time  = 0

    def dump_png(self, image):
        viz = self.viz
        try:
            _height, _width, channels = image.shape
            assert channels in [1, 3]
            assert image.dtype == np.uint8
            os.makedirs(self.path, exist_ok=True)
            file_id = 0
            for entry in os.scandir(self.path):
                if entry.is_file():
                    match = re.fullmatch(r'(\d+).*', entry.name)
                    if match:
                        file_id = max(file_id, int(match.group(1)) + 1)
            if channels == 1:
                pil_image = PIL.Image.fromarray(image[:, :, 0], 'L')
            else:
                pil_image = PIL.Image.fromarray(image, 'RGB')
            pil_image.save(os.path.join(self.path, f'{file_id:05d}.png'))
        except:
            viz.result.error = renderer.CapturedException()

    @imgui_utils.scoped_by_object_id
    def __call__(self, show=True):
        viz = self.viz
        if show:
            with imgui_utils.grayed_out(self.disabled_time != 0):
                imgui.text('Capture')
                imgui.same_line(viz.label_w)

                _changed, self.path = imgui_utils.input_text('##path', self.path, 1024,
                    flags=(imgui.INPUT_TEXT_AUTO_SELECT_ALL | imgui.INPUT_TEXT_ENTER_RETURNS_TRUE),
                    width=(-1),
                    help_text='PATH')
                if imgui.is_item_hovered() and not imgui.is_item_active() and self.path != '':
                    imgui.set_tooltip(self.path)
                imgui.text(' ')
                imgui.same_line(viz.label_w)
                if imgui_utils.button('Save image', width=viz.button_w, enabled=(self.disabled_time == 0 and 'image' in viz.result)):
                    self.dump_image = True
                    self.defer_frames = 2
                    self.disabled_time = 0.5
                imgui.same_line()
                if imgui_utils.button('Save GUI', width=viz.button_w, enabled=(self.disabled_time == 0)):
                    self.dump_gui = True
                    self.defer_frames = 2
                    self.disabled_time = 0.5

        self.disabled_time = max(self.disabled_time - viz.frame_delta, 0)
        if self.defer_frames > 0:
            self.defer_frames -= 1
        elif self.dump_image:
            if 'image' in viz.result:
                self.dump_png(viz.result.image)
            self.dump_image = False
        elif self.dump_gui:
            viz.capture_next_frame()
            self.dump_gui = False
        captured_frame = viz.pop_captured_frame()
        if captured_frame is not None:
            self.dump_png(captured_frame)

#----------------------------------------------------------------------------
