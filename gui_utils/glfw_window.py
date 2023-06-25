# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import time
import glfw
import OpenGL.GL as gl
from . import gl_utils

#----------------------------------------------------------------------------

class GlfwWindow: # pylint: disable=too-many-public-methods
    def __init__(self, *, title='GlfwWindow', window_width=1920, window_height=1080, deferred_show=True, close_on_esc=True):
        self._glfw_window           = None
        self._drawing_frame         = False
        self._frame_start_time      = None
        self._frame_delta           = 0
        self._fps_limit             = None
        self._vsync                 = None
        self._skip_frames           = 0
        self._deferred_show         = deferred_show
        self._close_on_esc          = close_on_esc
        self._esc_pressed           = False
        self._drag_and_drop_paths   = None
        self._capture_next_frame    = False
        self._captured_frame        = None

        # Create window.
        glfw.init()
        glfw.window_hint(glfw.VISIBLE, False)
        self._glfw_window = glfw.create_window(width=window_width, height=window_height, title=title, monitor=None, share=None)
        self._attach_glfw_callbacks()
        self.make_context_current()

        # Adjust window.
        self.set_vsync(False)
        self.set_window_size(window_width, window_height)
        if not self._deferred_show:
            glfw.show_window(self._glfw_window)

    def close(self):
        if self._drawing_frame:
            self.end_frame()
        if self._glfw_window is not None:
            glfw.destroy_window(self._glfw_window)
            self._glfw_window = None
        #glfw.terminate() # Commented out to play it nice with other glfw clients.

    def __del__(self):
        try:
            self.close()
        except:
            pass

    @property
    def window_width(self):
        return self.content_width

    @property
    def window_height(self):
        return self.content_height + self.title_bar_height

    @property
    def content_width(self):
        width, _height = glfw.get_window_size(self._glfw_window)
        return width

    @property
    def content_height(self):
        _width, height = glfw.get_window_size(self._glfw_window)
        return height

    @property
    def title_bar_height(self):
        _left, top, _right, _bottom = glfw.get_window_frame_size(self._glfw_window)
        return top

    @property
    def monitor_width(self):
        _, _, width, _height = glfw.get_monitor_workarea(glfw.get_primary_monitor())
        return width

    @property
    def monitor_height(self):
        _, _, _width, height = glfw.get_monitor_workarea(glfw.get_primary_monitor())
        return height

    @property
    def frame_delta(self):
        return self._frame_delta

    def set_title(self, title):
        glfw.set_window_title(self._glfw_window, title)

    def set_window_size(self, width, height):
        width = min(width, self.monitor_width)
        height = min(height, self.monitor_height)
        glfw.set_window_size(self._glfw_window, width, max(height - self.title_bar_height, 0))
        if width == self.monitor_width and height == self.monitor_height:
            self.maximize()

    def set_content_size(self, width, height):
        self.set_window_size(width, height + self.title_bar_height)

    def maximize(self):
        glfw.maximize_window(self._glfw_window)

    def set_position(self, x, y):
        glfw.set_window_pos(self._glfw_window, x, y + self.title_bar_height)

    def center(self):
        self.set_position((self.monitor_width - self.window_width) // 2, (self.monitor_height - self.window_height) // 2)

    def set_vsync(self, vsync):
        vsync = bool(vsync)
        if vsync != self._vsync:
            glfw.swap_interval(1 if vsync else 0)
            self._vsync = vsync

    def set_fps_limit(self, fps_limit):
        self._fps_limit = int(fps_limit)

    def should_close(self):
        return glfw.window_should_close(self._glfw_window) or (self._close_on_esc and self._esc_pressed)

    def skip_frame(self):
        self.skip_frames(1)

    def skip_frames(self, num): # Do not update window for the next N frames.
        self._skip_frames = max(self._skip_frames, int(num))

    def is_skipping_frames(self):
        return self._skip_frames > 0

    def capture_next_frame(self):
        self._capture_next_frame = True

    def pop_captured_frame(self):
        frame = self._captured_frame
        self._captured_frame = None
        return frame

    def pop_drag_and_drop_paths(self):
        paths = self._drag_and_drop_paths
        self._drag_and_drop_paths = None
        return paths

    def draw_frame(self): # To be overridden by subclass.
        self.begin_frame()
        # Rendering code goes here.
        self.end_frame()

    def make_context_current(self):
        if self._glfw_window is not None:
            glfw.make_context_current(self._glfw_window)

    def begin_frame(self):
        # End previous frame.
        if self._drawing_frame:
            self.end_frame()

        # Apply FPS limit.
        if self._frame_start_time is not None and self._fps_limit is not None:
            delay = self._frame_start_time - time.perf_counter() + 1 / self._fps_limit
            if delay > 0:
                time.sleep(delay)
        cur_time = time.perf_counter()
        if self._frame_start_time is not None:
            self._frame_delta = cur_time - self._frame_start_time
        self._frame_start_time = cur_time

        # Process events.
        glfw.poll_events()

        # Begin frame.
        self._drawing_frame = True
        self.make_context_current()

        # Initialize GL state.
        gl.glViewport(0, 0, self.content_width, self.content_height)
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadIdentity()
        gl.glTranslate(-1, 1, 0)
        gl.glScale(2 / max(self.content_width, 1), -2 / max(self.content_height, 1), 1)
        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glLoadIdentity()
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_ONE, gl.GL_ONE_MINUS_SRC_ALPHA) # Pre-multiplied alpha.

        # Clear.
        gl.glClearColor(0, 0, 0, 1)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

    def end_frame(self):
        assert self._drawing_frame
        self._drawing_frame = False

        # Skip frames if requested.
        if self._skip_frames > 0:
            self._skip_frames -= 1
            return

        # Capture frame if requested.
        if self._capture_next_frame:
            self._captured_frame = gl_utils.read_pixels(self.content_width, self.content_height)
            self._capture_next_frame = False

        # Update window.
        if self._deferred_show:
            glfw.show_window(self._glfw_window)
            self._deferred_show = False
        glfw.swap_buffers(self._glfw_window)

    def _attach_glfw_callbacks(self):
        glfw.set_key_callback(self._glfw_window, self._glfw_key_callback)
        glfw.set_drop_callback(self._glfw_window, self._glfw_drop_callback)

    def _glfw_key_callback(self, _window, key, _scancode, action, _mods):
        if action == glfw.PRESS and key == glfw.KEY_ESCAPE:
            self._esc_pressed = True

    def _glfw_drop_callback(self, _window, paths):
        self._drag_and_drop_paths = paths

#----------------------------------------------------------------------------
