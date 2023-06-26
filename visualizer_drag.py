# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import click
import os

import multiprocessing
import numpy as np
import torch
import imgui
import dnnlib
from gui_utils import imgui_window
from gui_utils import imgui_utils
from gui_utils import gl_utils
from gui_utils import text_utils
from viz import renderer
from viz import pickle_widget
from viz import latent_widget
from viz import drag_widget
from viz import capture_widget

#----------------------------------------------------------------------------

class Visualizer(imgui_window.ImguiWindow):
    def __init__(self, capture_dir=None):
        super().__init__(title='DragGAN', window_width=3840, window_height=2160)

        # Internals.
        self._last_error_print  = None
        self._async_renderer    = AsyncRenderer()
        self._defer_rendering   = 0
        self._tex_img           = None
        self._tex_obj           = None
        self._mask_obj          = None
        self._image_area        = None
        self._status            = dnnlib.EasyDict()

        # Widget interface.
        self.args               = dnnlib.EasyDict()
        self.result             = dnnlib.EasyDict()
        self.pane_w             = 0
        self.label_w            = 0
        self.button_w           = 0
        self.image_w            = 0
        self.image_h            = 0

        # Widgets.
        self.pickle_widget      = pickle_widget.PickleWidget(self)
        self.latent_widget      = latent_widget.LatentWidget(self)
        self.drag_widget        = drag_widget.DragWidget(self)
        self.capture_widget     = capture_widget.CaptureWidget(self)

        if capture_dir is not None:
            self.capture_widget.path = capture_dir

        # Initialize window.
        self.set_position(0, 0)
        self._adjust_font_size()
        self.skip_frame() # Layout may change after first frame.

    def close(self):
        super().close()
        if self._async_renderer is not None:
            self._async_renderer.close()
            self._async_renderer = None

    def add_recent_pickle(self, pkl, ignore_errors=False):
        self.pickle_widget.add_recent(pkl, ignore_errors=ignore_errors)

    def load_pickle(self, pkl, ignore_errors=False):
        self.pickle_widget.load(pkl, ignore_errors=ignore_errors)

    def print_error(self, error):
        error = str(error)
        if error != self._last_error_print:
            print('\n' + error + '\n')
            self._last_error_print = error

    def defer_rendering(self, num_frames=1):
        self._defer_rendering = max(self._defer_rendering, num_frames)

    def clear_result(self):
        self._async_renderer.clear_result()

    def set_async(self, is_async):
        if is_async != self._async_renderer.is_async:
            self._async_renderer.set_async(is_async)
            self.clear_result()
            if 'image' in self.result:
                self.result.message = 'Switching rendering process...'
                self.defer_rendering()

    def _adjust_font_size(self):
        old = self.font_size
        self.set_font_size(min(self.content_width / 120, self.content_height / 60))
        if self.font_size != old:
            self.skip_frame() # Layout changed.

    def check_update_mask(self, **args):
        update_mask = False
        if 'pkl' in self._status:
            if self._status.pkl != args['pkl']:
                update_mask = True
        self._status.pkl = args['pkl']
        if 'w0_seed' in self._status:
            if self._status.w0_seed != args['w0_seed']:
                update_mask = True
        self._status.w0_seed = args['w0_seed']
        return update_mask

    def capture_image_frame(self):
        self.capture_next_frame()
        captured_frame = self.pop_captured_frame()
        captured_image = None
        if captured_frame is not None:
            x1, y1, w, h = self._image_area
            captured_image = captured_frame[y1:y1+h, x1:x1+w, :]
        return captured_image

    def get_drag_info(self):
        seed = self.latent_widget.seed
        points = self.drag_widget.points
        targets = self.drag_widget.targets
        mask = self.drag_widget.mask
        w = self._async_renderer._renderer_obj.w
        return seed, points, targets, mask, w

    def draw_frame(self):
        self.begin_frame()
        self.args = dnnlib.EasyDict()
        self.pane_w = self.font_size * 18
        self.button_w = self.font_size * 5
        self.label_w = round(self.font_size * 4.5)

        # Detect mouse dragging in the result area.
        if self._image_area is not None:
            if not hasattr(self.drag_widget, 'width'):
                self.drag_widget.init_mask(self.image_w, self.image_h)
            clicked, down, img_x, img_y = imgui_utils.click_hidden_window(
                '##image_area', self._image_area[0], self._image_area[1], self._image_area[2], self._image_area[3], self.image_w, self.image_h)
            self.drag_widget.action(clicked, down, img_x, img_y)

        # Begin control pane.
        imgui.set_next_window_position(0, 0)
        imgui.set_next_window_size(self.pane_w, self.content_height)
        imgui.begin('##control_pane', closable=False, flags=(imgui.WINDOW_NO_TITLE_BAR | imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_MOVE))

        # Widgets.
        expanded, _visible = imgui_utils.collapsing_header('Network & latent', default=True)
        self.pickle_widget(expanded)
        self.latent_widget(expanded)
        expanded, _visible = imgui_utils.collapsing_header('Drag', default=True)
        self.drag_widget(expanded)
        expanded, _visible = imgui_utils.collapsing_header('Capture', default=True)
        self.capture_widget(expanded)

        # Render.
        if self.is_skipping_frames():
            pass
        elif self._defer_rendering > 0:
            self._defer_rendering -= 1
        elif self.args.pkl is not None:
            self._async_renderer.set_args(**self.args)
            result = self._async_renderer.get_result()
            if result is not None:
                self.result = result        
                if 'stop' in self.result and self.result.stop:
                    self.drag_widget.stop_drag()
                if 'points' in self.result:
                    self.drag_widget.set_points(self.result.points)
                if 'init_net' in self.result:
                    if self.result.init_net:
                        self.drag_widget.reset_point()

        # Display.
        max_w = self.content_width - self.pane_w
        max_h = self.content_height
        pos = np.array([self.pane_w + max_w / 2, max_h / 2])
        if 'image' in self.result:
            # Reset mask after loading a new pickle or changing seed.
            if self.check_update_mask(**self.args):
                h, w, _ = self.result.image.shape
                self.drag_widget.init_mask(w, h)

            if self._tex_img is not self.result.image:
                self._tex_img = self.result.image
                if self._tex_obj is None or not self._tex_obj.is_compatible(image=self._tex_img):
                    self._tex_obj = gl_utils.Texture(image=self._tex_img, bilinear=False, mipmap=False)
                else:
                    self._tex_obj.update(self._tex_img)
                self.image_h, self.image_w = self._tex_obj.height, self._tex_obj.width
            zoom = min(max_w / self._tex_obj.width, max_h / self._tex_obj.height)
            zoom = np.floor(zoom) if zoom >= 1 else zoom
            self._tex_obj.draw(pos=pos, zoom=zoom, align=0.5, rint=True)
            if self.drag_widget.show_mask and hasattr(self.drag_widget, 'mask'):
                mask = ((1-self.drag_widget.mask.unsqueeze(-1)) * 255).to(torch.uint8)
                if self._mask_obj is None or not self._mask_obj.is_compatible(image=self._tex_img):
                    self._mask_obj = gl_utils.Texture(image=mask, bilinear=False, mipmap=False)
                else:
                    self._mask_obj.update(mask)
                self._mask_obj.draw(pos=pos, zoom=zoom, align=0.5, rint=True, alpha=0.15)

            if self.drag_widget.mode in ['flexible', 'fixed']:
                posx, posy = imgui.get_mouse_pos()
                if posx >= self.pane_w:
                    pos_c = np.array([posx, posy])
                    gl_utils.draw_circle(center=pos_c, radius=self.drag_widget.r_mask * zoom, alpha=0.5)
            
            rescale = self._tex_obj.width / 512 * zoom
            
            for point in self.drag_widget.targets:
                pos_x = self.pane_w + max_w / 2 + (point[1] - self.image_w//2) * zoom
                pos_y = max_h / 2 + (point[0] - self.image_h//2) * zoom
                gl_utils.draw_circle(center=np.array([pos_x, pos_y]), color=[0,0,1], radius=9 * rescale)
            
            for point in self.drag_widget.points:
                pos_x = self.pane_w + max_w / 2 + (point[1] - self.image_w//2) * zoom
                pos_y = max_h / 2 + (point[0] - self.image_h//2) * zoom
                gl_utils.draw_circle(center=np.array([pos_x, pos_y]), color=[1,0,0], radius=9 * rescale)

            for point, target in zip(self.drag_widget.points, self.drag_widget.targets):
                t_x = self.pane_w + max_w / 2 + (target[1] - self.image_w//2) * zoom
                t_y = max_h / 2 + (target[0] - self.image_h//2) * zoom

                p_x = self.pane_w + max_w / 2 + (point[1] - self.image_w//2) * zoom
                p_y = max_h / 2 + (point[0] - self.image_h//2) * zoom

                gl_utils.draw_arrow(p_x, p_y, t_x, t_y, l=8 * rescale, width = 3 * rescale)

            imshow_w = int(self._tex_obj.width * zoom)
            imshow_h = int(self._tex_obj.height * zoom)
            self._image_area = [int(self.pane_w + max_w / 2 - imshow_w / 2), int(max_h / 2 - imshow_h / 2), imshow_w, imshow_h]
        if 'error' in self.result:
            self.print_error(self.result.error)
            if 'message' not in self.result:
                self.result.message = str(self.result.error)
        if 'message' in self.result:
            tex = text_utils.get_texture(self.result.message, size=self.font_size, max_width=max_w, max_height=max_h, outline=2)
            tex.draw(pos=pos, align=0.5, rint=True, color=1)

        # End frame.
        self._adjust_font_size()
        imgui.end()
        self.end_frame()

#----------------------------------------------------------------------------

class AsyncRenderer:
    def __init__(self):
        self._closed        = False
        self._is_async      = False
        self._cur_args      = None
        self._cur_result    = None
        self._cur_stamp     = 0
        self._renderer_obj  = None
        self._args_queue    = None
        self._result_queue  = None
        self._process       = None

    def close(self):
        self._closed = True
        self._renderer_obj = None
        if self._process is not None:
            self._process.terminate()
        self._process = None
        self._args_queue = None
        self._result_queue = None

    @property
    def is_async(self):
        return self._is_async

    def set_async(self, is_async):
        self._is_async = is_async

    def set_args(self, **args):
        assert not self._closed
        args2 = args.copy()
        args_mask = args2.pop('mask')
        if self._cur_args:
            _cur_args = self._cur_args.copy()
            cur_args_mask = _cur_args.pop('mask')
        else:
            _cur_args = self._cur_args
        # if args != self._cur_args:
        if args2 != _cur_args:
            if self._is_async:
                self._set_args_async(**args)
            else:
                self._set_args_sync(**args)
            self._cur_args = args

    def _set_args_async(self, **args):
        if self._process is None:
            self._args_queue = multiprocessing.Queue()
            self._result_queue = multiprocessing.Queue()
            try:
                multiprocessing.set_start_method('spawn')
            except RuntimeError:
                pass
            self._process = multiprocessing.Process(target=self._process_fn, args=(self._args_queue, self._result_queue), daemon=True)
            self._process.start()
        self._args_queue.put([args, self._cur_stamp])

    def _set_args_sync(self, **args):
        if self._renderer_obj is None:
            self._renderer_obj = renderer.Renderer()
        self._cur_result = self._renderer_obj.render(**args)

    def get_result(self):
        assert not self._closed
        if self._result_queue is not None:
            while self._result_queue.qsize() > 0:
                result, stamp = self._result_queue.get()
                if stamp == self._cur_stamp:
                    self._cur_result = result
        return self._cur_result

    def clear_result(self):
        assert not self._closed
        self._cur_args = None
        self._cur_result = None
        self._cur_stamp += 1

    @staticmethod
    def _process_fn(args_queue, result_queue):
        renderer_obj = renderer.Renderer()
        cur_args = None
        cur_stamp = None
        while True:
            args, stamp = args_queue.get()
            while args_queue.qsize() > 0:
                args, stamp = args_queue.get()
            if args != cur_args or stamp != cur_stamp:
                result = renderer_obj.render(**args)
                if 'error' in result:
                    result.error = renderer.CapturedException(result.error)
                result_queue.put([result, stamp])
                cur_args = args
                cur_stamp = stamp

#----------------------------------------------------------------------------

@click.command()
@click.argument('pkls', metavar='PATH', nargs=-1)
@click.option('--capture-dir', help='Where to save screenshot captures', metavar='PATH', default=None)
@click.option('--browse-dir', help='Specify model path for the \'Browse...\' button', metavar='PATH')
def main(
    pkls,
    capture_dir,
    browse_dir
):
    """Interactive model visualizer.

    Optional PATH argument can be used specify which .pkl file to load.
    """
    viz = Visualizer(capture_dir=capture_dir)

    if browse_dir is not None:
        viz.pickle_widget.search_dirs = [browse_dir]

    # List pickles.
    if len(pkls) > 0:
        for pkl in pkls:
            viz.add_recent_pickle(pkl)
        viz.load_pickle(pkls[0])
    else:
        pretrained = [
            'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan2/versions/1/files/stylegan2-afhqcat-512x512.pkl',
            'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan2/versions/1/files/stylegan2-afhqdog-512x512.pkl',
            'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan2/versions/1/files/stylegan2-afhqv2-512x512.pkl',
            'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan2/versions/1/files/stylegan2-afhqwild-512x512.pkl',
            'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan2/versions/1/files/stylegan2-brecahad-512x512.pkl',
            'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan2/versions/1/files/stylegan2-celebahq-256x256.pkl',
            'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan2/versions/1/files/stylegan2-cifar10-32x32.pkl',
            'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan2/versions/1/files/stylegan2-ffhq-1024x1024.pkl',
            'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan2/versions/1/files/stylegan2-ffhq-256x256.pkl',
            'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan2/versions/1/files/stylegan2-ffhq-512x512.pkl',
            'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan2/versions/1/files/stylegan2-ffhqu-1024x1024.pkl',
            'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan2/versions/1/files/stylegan2-ffhqu-256x256.pkl',
            'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan2/versions/1/files/stylegan2-lsundog-256x256.pkl',
            'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan2/versions/1/files/stylegan2-metfaces-1024x1024.pkl',
            'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan2/versions/1/files/stylegan2-metfacesu-1024x1024.pkl'
        ]

        # Populate recent pickles list with pretrained model URLs.
        for url in pretrained:
            viz.add_recent_pickle(url)

    # Run.
    while not viz.should_close():
        viz.draw_frame()
    viz.close()

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
