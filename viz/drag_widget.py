import os
import torch
import numpy as np
import imgui
import dnnlib
from gui_utils import imgui_utils

#----------------------------------------------------------------------------

class DragWidget:
    def __init__(self, viz):
        self.viz            = viz
        self.point          = [-1, -1]
        self.points         = []
        self.targets        = []
        self.is_point       = True
        self.last_click     = False
        self.is_drag        = False
        self.iteration      = 0
        self.mode           = 'point'
        self.r_mask         = 50
        self.show_mask      = False
        self.mask           = torch.ones(256, 256)
        self.lambda_mask    = 20
        self.feature_idx    = 5
        self.r1             = 3
        self.r2             = 12
        self.path           = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '_screenshots'))
        self.defer_frames   = 0
        self.disabled_time  = 0

    def action(self, click, down, x, y):
        if self.mode == 'point':
            self.add_point(click, x, y)
        elif down:
            self.draw_mask(x, y)

    def add_point(self, click, x, y):
        if click:
            self.point = [y, x]
        elif self.last_click:
            if self.is_drag:
                self.stop_drag()
            if self.is_point:
                self.points.append(self.point)
                self.is_point = False
            else:
                self.targets.append(self.point)
                self.is_point = True
        self.last_click = click

    def init_mask(self, w, h):
        self.width, self.height = w, h
        self.mask = torch.ones(h, w)

    def draw_mask(self, x, y):
        X = torch.linspace(0, self.width, self.width)
        Y = torch.linspace(0, self.height, self.height)
        yy, xx = torch.meshgrid(Y, X)
        circle = (xx - x)**2 + (yy - y)**2 < self.r_mask**2
        if self.mode == 'flexible':
            self.mask[circle] = 0
        elif self.mode == 'fixed':
            self.mask[circle] = 1

    def stop_drag(self):
        self.is_drag = False
        self.iteration = 0

    def set_points(self, points):
        self.points = points

    def reset_point(self):
        self.points = []
        self.targets = []
        self.is_point = True

    def load_points(self, suffix):
        points = []
        point_path = self.path + f'_{suffix}.txt'
        try:
            with open(point_path, "r") as f:
                for line in f.readlines():
                    y, x = line.split()
                    points.append([int(y), int(x)])
        except:
            print(f'Wrong point file path: {point_path}')
        return points

    @imgui_utils.scoped_by_object_id
    def __call__(self, show=True):
        viz = self.viz
        if show:
            with imgui_utils.grayed_out(self.disabled_time != 0):
                imgui.text('Drag')
                imgui.same_line(viz.label_w)

                if imgui_utils.button('Add point', width=viz.button_w, enabled='image' in viz.result):
                    self.mode = 'point'

                imgui.same_line()
                reset = False
                if imgui_utils.button('Reset point', width=viz.button_w, enabled='image' in viz.result):
                    self.reset_point()
                    reset = True

                imgui.text(' ')
                imgui.same_line(viz.label_w)
                if imgui_utils.button('Start', width=viz.button_w, enabled='image' in viz.result):
                    self.is_drag = True
                    if len(self.points) > len(self.targets):
                        self.points = self.points[:len(self.targets)]

                imgui.same_line()
                if imgui_utils.button('Stop', width=viz.button_w, enabled='image' in viz.result):
                    self.stop_drag()

                imgui.text(' ')
                imgui.same_line(viz.label_w)
                imgui.text(f'Steps: {self.iteration}')
                
                imgui.text('Mask')
                imgui.same_line(viz.label_w)
                if imgui_utils.button('Flexible area', width=viz.button_w, enabled='image' in viz.result):
                    self.mode = 'flexible'
                    self.show_mask = True
                
                imgui.same_line()
                if imgui_utils.button('Fixed area', width=viz.button_w, enabled='image' in viz.result):
                    self.mode = 'fixed'
                    self.show_mask = True
                
                imgui.text(' ')
                imgui.same_line(viz.label_w)
                if imgui_utils.button('Reset mask', width=viz.button_w, enabled='image' in viz.result):
                    self.mask = torch.ones(self.height, self.width)
                imgui.same_line()
                _clicked, self.show_mask = imgui.checkbox('Show mask', self.show_mask)

                imgui.text(' ')
                imgui.same_line(viz.label_w)
                with imgui_utils.item_width(viz.font_size * 6):
                    changed, self.r_mask = imgui.input_int('Radius', self.r_mask)

                imgui.text(' ')
                imgui.same_line(viz.label_w)
                with imgui_utils.item_width(viz.font_size * 6):
                    changed, self.lambda_mask = imgui.input_int('Lambda', self.lambda_mask)

        self.disabled_time = max(self.disabled_time - viz.frame_delta, 0)
        if self.defer_frames > 0:
            self.defer_frames -= 1
        viz.args.is_drag = self.is_drag
        if self.is_drag:
            self.iteration += 1
        viz.args.iteration = self.iteration
        viz.args.points = [point for point in self.points]
        viz.args.targets = [point for point in self.targets]
        viz.args.mask = self.mask
        viz.args.lambda_mask = self.lambda_mask
        viz.args.feature_idx = self.feature_idx
        viz.args.r1 = self.r1
        viz.args.r2 = self.r2
        viz.args.reset = reset


#----------------------------------------------------------------------------
