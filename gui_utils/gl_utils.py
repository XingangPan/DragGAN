# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import math
import os
import functools
import contextlib
import numpy as np
import OpenGL.GL as gl
import OpenGL.GL.ARB.texture_float
import dnnlib

#----------------------------------------------------------------------------

def init_egl():
    assert os.environ['PYOPENGL_PLATFORM'] == 'egl' # Must be set before importing OpenGL.
    import OpenGL.EGL as egl
    import ctypes

    # Initialize EGL.
    display = egl.eglGetDisplay(egl.EGL_DEFAULT_DISPLAY)
    assert display != egl.EGL_NO_DISPLAY
    major = ctypes.c_int32()
    minor = ctypes.c_int32()
    ok = egl.eglInitialize(display, major, minor)
    assert ok
    assert major.value * 10 + minor.value >= 14

    # Choose config.
    config_attribs = [
        egl.EGL_RENDERABLE_TYPE,    egl.EGL_OPENGL_BIT,
        egl.EGL_SURFACE_TYPE,       egl.EGL_PBUFFER_BIT,
        egl.EGL_NONE
    ]
    configs = (ctypes.c_int32 * 1)()
    num_configs = ctypes.c_int32()
    ok = egl.eglChooseConfig(display, config_attribs, configs, 1, num_configs)
    assert ok
    assert num_configs.value == 1
    config = configs[0]

    # Create dummy pbuffer surface.
    surface_attribs = [
        egl.EGL_WIDTH,  1,
        egl.EGL_HEIGHT, 1,
        egl.EGL_NONE
    ]
    surface = egl.eglCreatePbufferSurface(display, config, surface_attribs)
    assert surface != egl.EGL_NO_SURFACE

    # Setup GL context.
    ok = egl.eglBindAPI(egl.EGL_OPENGL_API)
    assert ok
    context = egl.eglCreateContext(display, config, egl.EGL_NO_CONTEXT, None)
    assert context != egl.EGL_NO_CONTEXT
    ok = egl.eglMakeCurrent(display, surface, surface, context)
    assert ok

#----------------------------------------------------------------------------

_texture_formats = {
    ('uint8',   1): dnnlib.EasyDict(type=gl.GL_UNSIGNED_BYTE, format=gl.GL_LUMINANCE,       internalformat=gl.GL_LUMINANCE8),
    ('uint8',   2): dnnlib.EasyDict(type=gl.GL_UNSIGNED_BYTE, format=gl.GL_LUMINANCE_ALPHA, internalformat=gl.GL_LUMINANCE8_ALPHA8),
    ('uint8',   3): dnnlib.EasyDict(type=gl.GL_UNSIGNED_BYTE, format=gl.GL_RGB,             internalformat=gl.GL_RGB8),
    ('uint8',   4): dnnlib.EasyDict(type=gl.GL_UNSIGNED_BYTE, format=gl.GL_RGBA,            internalformat=gl.GL_RGBA8),
    ('float32', 1): dnnlib.EasyDict(type=gl.GL_FLOAT,         format=gl.GL_LUMINANCE,       internalformat=OpenGL.GL.ARB.texture_float.GL_LUMINANCE32F_ARB),
    ('float32', 2): dnnlib.EasyDict(type=gl.GL_FLOAT,         format=gl.GL_LUMINANCE_ALPHA, internalformat=OpenGL.GL.ARB.texture_float.GL_LUMINANCE_ALPHA32F_ARB),
    ('float32', 3): dnnlib.EasyDict(type=gl.GL_FLOAT,         format=gl.GL_RGB,             internalformat=gl.GL_RGB32F),
    ('float32', 4): dnnlib.EasyDict(type=gl.GL_FLOAT,         format=gl.GL_RGBA,            internalformat=gl.GL_RGBA32F),
}

def get_texture_format(dtype, channels):
    return _texture_formats[(np.dtype(dtype).name, int(channels))]

#----------------------------------------------------------------------------

def prepare_texture_data(image):
    image = np.asarray(image)
    if image.ndim == 2:
        image = image[:, :, np.newaxis]
    if image.dtype.name == 'float64':
        image = image.astype('float32')
    return image

#----------------------------------------------------------------------------

def draw_pixels(image, *, pos=0, zoom=1, align=0, rint=True):
    pos = np.broadcast_to(np.asarray(pos, dtype='float32'), [2])
    zoom = np.broadcast_to(np.asarray(zoom, dtype='float32'), [2])
    align = np.broadcast_to(np.asarray(align, dtype='float32'), [2])
    image = prepare_texture_data(image)
    height, width, channels = image.shape
    size = zoom * [width, height]
    pos = pos - size * align
    if rint:
        pos = np.rint(pos)
    fmt = get_texture_format(image.dtype, channels)

    gl.glPushAttrib(gl.GL_CURRENT_BIT | gl.GL_PIXEL_MODE_BIT)
    gl.glPushClientAttrib(gl.GL_CLIENT_PIXEL_STORE_BIT)
    gl.glRasterPos2f(pos[0], pos[1])
    gl.glPixelZoom(zoom[0], -zoom[1])
    gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 1)
    gl.glDrawPixels(width, height, fmt.format, fmt.type, image)
    gl.glPopClientAttrib()
    gl.glPopAttrib()

#----------------------------------------------------------------------------

def read_pixels(width, height, *, pos=0, dtype='uint8', channels=3):
    pos = np.broadcast_to(np.asarray(pos, dtype='float32'), [2])
    dtype = np.dtype(dtype)
    fmt = get_texture_format(dtype, channels)
    image = np.empty([height, width, channels], dtype=dtype)

    gl.glPushClientAttrib(gl.GL_CLIENT_PIXEL_STORE_BIT)
    gl.glPixelStorei(gl.GL_PACK_ALIGNMENT, 1)
    gl.glReadPixels(int(np.round(pos[0])), int(np.round(pos[1])), width, height, fmt.format, fmt.type, image)
    gl.glPopClientAttrib()
    return np.flipud(image)

#----------------------------------------------------------------------------

class Texture:
    def __init__(self, *, image=None, width=None, height=None, channels=None, dtype=None, bilinear=True, mipmap=True):
        self.gl_id = None
        self.bilinear = bilinear
        self.mipmap = mipmap

        # Determine size and dtype.
        if image is not None:
            image = prepare_texture_data(image)
            self.height, self.width, self.channels = image.shape
            self.dtype = image.dtype
        else:
            assert width is not None and height is not None
            self.width = width
            self.height = height
            self.channels = channels if channels is not None else 3
            self.dtype = np.dtype(dtype) if dtype is not None else np.uint8

        # Validate size and dtype.
        assert isinstance(self.width, int) and self.width >= 0
        assert isinstance(self.height, int) and self.height >= 0
        assert isinstance(self.channels, int) and self.channels >= 1
        assert self.is_compatible(width=width, height=height, channels=channels, dtype=dtype)

        # Create texture object.
        self.gl_id = gl.glGenTextures(1)
        with self.bind():
            gl.glTexParameterf(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
            gl.glTexParameterf(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)
            gl.glTexParameterf(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR if self.bilinear else gl.GL_NEAREST)
            gl.glTexParameterf(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR_MIPMAP_LINEAR if self.mipmap else gl.GL_NEAREST)
        self.update(image)

    def delete(self):
        if self.gl_id is not None:
            gl.glDeleteTextures([self.gl_id])
            self.gl_id = None

    def __del__(self):
        try:
            self.delete()
        except:
            pass

    @contextlib.contextmanager
    def bind(self):
        prev_id = gl.glGetInteger(gl.GL_TEXTURE_BINDING_2D)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.gl_id)
        yield
        gl.glBindTexture(gl.GL_TEXTURE_2D, prev_id)

    def update(self, image):
        if image is not None:
            image = prepare_texture_data(image)
            assert self.is_compatible(image=image)
        with self.bind():
            fmt = get_texture_format(self.dtype, self.channels)
            gl.glPushClientAttrib(gl.GL_CLIENT_PIXEL_STORE_BIT)
            gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 1)
            gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, fmt.internalformat, self.width, self.height, 0, fmt.format, fmt.type, image)
            if self.mipmap:
                gl.glGenerateMipmap(gl.GL_TEXTURE_2D)
            gl.glPopClientAttrib()

    def draw(self, *, pos=0, zoom=1, align=0, rint=False, color=1, alpha=1, rounding=0):
        zoom = np.broadcast_to(np.asarray(zoom, dtype='float32'), [2])
        size = zoom * [self.width, self.height]
        with self.bind():
            gl.glPushAttrib(gl.GL_ENABLE_BIT)
            gl.glEnable(gl.GL_TEXTURE_2D)
            draw_rect(pos=pos, size=size, align=align, rint=rint, color=color, alpha=alpha, rounding=rounding)
            gl.glPopAttrib()

    def is_compatible(self, *, image=None, width=None, height=None, channels=None, dtype=None): # pylint: disable=too-many-return-statements
        if image is not None:
            if image.ndim != 3:
                return False
            ih, iw, ic = image.shape
            if not self.is_compatible(width=iw, height=ih, channels=ic, dtype=image.dtype):
                return False
        if width is not None and self.width != width:
            return False
        if height is not None and self.height != height:
            return False
        if channels is not None and self.channels != channels:
            return False
        if dtype is not None and self.dtype != dtype:
            return False
        return True

#----------------------------------------------------------------------------

class Framebuffer:
    def __init__(self, *, texture=None, width=None, height=None, channels=None, dtype=None, msaa=0):
        self.texture = texture
        self.gl_id = None
        self.gl_color = None
        self.gl_depth_stencil = None
        self.msaa = msaa

        # Determine size and dtype.
        if texture is not None:
            assert isinstance(self.texture, Texture)
            self.width = texture.width
            self.height = texture.height
            self.channels = texture.channels
            self.dtype = texture.dtype
        else:
            assert width is not None and height is not None
            self.width = width
            self.height = height
            self.channels = channels if channels is not None else 4
            self.dtype = np.dtype(dtype) if dtype is not None else np.float32

        # Validate size and dtype.
        assert isinstance(self.width, int) and self.width >= 0
        assert isinstance(self.height, int) and self.height >= 0
        assert isinstance(self.channels, int) and self.channels >= 1
        assert width is None or width == self.width
        assert height is None or height == self.height
        assert channels is None or channels == self.channels
        assert dtype is None or dtype == self.dtype

        # Create framebuffer object.
        self.gl_id = gl.glGenFramebuffers(1)
        with self.bind():

            # Setup color buffer.
            if self.texture is not None:
                assert self.msaa == 0
                gl.glFramebufferTexture2D(gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0, gl.GL_TEXTURE_2D, self.texture.gl_id, 0)
            else:
                fmt = get_texture_format(self.dtype, self.channels)
                self.gl_color = gl.glGenRenderbuffers(1)
                gl.glBindRenderbuffer(gl.GL_RENDERBUFFER, self.gl_color)
                gl.glRenderbufferStorageMultisample(gl.GL_RENDERBUFFER, self.msaa, fmt.internalformat, self.width, self.height)
                gl.glFramebufferRenderbuffer(gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0, gl.GL_RENDERBUFFER, self.gl_color)

            # Setup depth/stencil buffer.
            self.gl_depth_stencil = gl.glGenRenderbuffers(1)
            gl.glBindRenderbuffer(gl.GL_RENDERBUFFER, self.gl_depth_stencil)
            gl.glRenderbufferStorageMultisample(gl.GL_RENDERBUFFER, self.msaa, gl.GL_DEPTH24_STENCIL8, self.width, self.height)
            gl.glFramebufferRenderbuffer(gl.GL_FRAMEBUFFER, gl.GL_DEPTH_STENCIL_ATTACHMENT, gl.GL_RENDERBUFFER, self.gl_depth_stencil)

    def delete(self):
        if self.gl_id is not None:
            gl.glDeleteFramebuffers([self.gl_id])
            self.gl_id = None
        if self.gl_color is not None:
            gl.glDeleteRenderbuffers(1, [self.gl_color])
            self.gl_color = None
        if self.gl_depth_stencil is not None:
            gl.glDeleteRenderbuffers(1, [self.gl_depth_stencil])
            self.gl_depth_stencil = None

    def __del__(self):
        try:
            self.delete()
        except:
            pass

    @contextlib.contextmanager
    def bind(self):
        prev_fbo = gl.glGetInteger(gl.GL_FRAMEBUFFER_BINDING)
        prev_rbo = gl.glGetInteger(gl.GL_RENDERBUFFER_BINDING)
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.gl_id)
        if self.width is not None and self.height is not None:
            gl.glViewport(0, 0, self.width, self.height)
        yield
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, prev_fbo)
        gl.glBindRenderbuffer(gl.GL_RENDERBUFFER, prev_rbo)

    def blit(self, dst=None):
        assert dst is None or isinstance(dst, Framebuffer)
        with self.bind():
            gl.glBindFramebuffer(gl.GL_DRAW_FRAMEBUFFER, 0 if dst is None else dst.fbo)
            gl.glBlitFramebuffer(0, 0, self.width, self.height, 0, 0, self.width, self.height, gl.GL_COLOR_BUFFER_BIT, gl.GL_NEAREST)

#----------------------------------------------------------------------------

def draw_shape(vertices, *, mode=gl.GL_TRIANGLE_FAN, pos=0, size=1, color=1, alpha=1):
    assert vertices.ndim == 2 and vertices.shape[1] == 2
    pos = np.broadcast_to(np.asarray(pos, dtype='float32'), [2])
    size = np.broadcast_to(np.asarray(size, dtype='float32'), [2])
    color = np.broadcast_to(np.asarray(color, dtype='float32'), [3])
    alpha = np.clip(np.broadcast_to(np.asarray(alpha, dtype='float32'), []), 0, 1)

    gl.glPushClientAttrib(gl.GL_CLIENT_VERTEX_ARRAY_BIT)
    gl.glPushAttrib(gl.GL_CURRENT_BIT | gl.GL_TRANSFORM_BIT)
    gl.glMatrixMode(gl.GL_MODELVIEW)
    gl.glPushMatrix()

    gl.glEnableClientState(gl.GL_VERTEX_ARRAY)
    gl.glEnableClientState(gl.GL_TEXTURE_COORD_ARRAY)
    gl.glVertexPointer(2, gl.GL_FLOAT, 0, vertices)
    gl.glTexCoordPointer(2, gl.GL_FLOAT, 0, vertices)
    gl.glTranslate(pos[0], pos[1], 0)
    gl.glScale(size[0], size[1], 1)
    gl.glColor4f(color[0] * alpha, color[1] * alpha, color[2] * alpha, alpha)
    gl.glDrawArrays(mode, 0, vertices.shape[0])

    gl.glPopMatrix()
    gl.glPopAttrib()
    gl.glPopClientAttrib()

#----------------------------------------------------------------------------

def draw_arrow(x1, y1, x2, y2, l=10, width=1.0):
    # Compute the length and angle of the arrow
    dx = x2 - x1
    dy = y2 - y1
    length = math.sqrt(dx**2 + dy**2)
    if length < l:
        return
    angle = math.atan2(dy, dx)
    
    # Save the current modelview matrix
    gl.glPushMatrix()
    
    # Translate and rotate the coordinate system
    gl.glTranslatef(x1, y1, 0.0)
    gl.glRotatef(angle * 180.0 / math.pi, 0.0, 0.0, 1.0)
    
    # Set the line width
    gl.glLineWidth(width)
    # gl.glColor3f(0.75, 0.75, 0.75)
    
    # Begin drawing lines
    gl.glBegin(gl.GL_LINES)
    
    # Draw the shaft of the arrow
    gl.glVertex2f(0.0, 0.0)
    gl.glVertex2f(length, 0.0)
    
    # Draw the head of the arrow
    gl.glVertex2f(length, 0.0)
    gl.glVertex2f(length - 2 * l, l)
    gl.glVertex2f(length, 0.0)
    gl.glVertex2f(length - 2 * l, -l)
    
    # End drawing lines
    gl.glEnd()
    
    # Restore the modelview matrix
    gl.glPopMatrix()

#----------------------------------------------------------------------------

def draw_rect(*, pos=0, pos2=None, size=None, align=0, rint=False, color=1, alpha=1, rounding=0):
    assert pos2 is None or size is None
    pos = np.broadcast_to(np.asarray(pos, dtype='float32'), [2])
    pos2 = np.broadcast_to(np.asarray(pos2, dtype='float32'), [2]) if pos2 is not None else None
    size = np.broadcast_to(np.asarray(size, dtype='float32'), [2]) if size is not None else None
    size = size if size is not None else pos2 - pos if pos2 is not None else np.array([1, 1], dtype='float32')
    pos = pos - size * align
    if rint:
        pos = np.rint(pos)
    rounding = np.broadcast_to(np.asarray(rounding, dtype='float32'), [2])
    rounding = np.minimum(np.abs(rounding) / np.maximum(np.abs(size), 1e-8), 0.5)
    if np.min(rounding) == 0:
        rounding *= 0
    vertices = _setup_rect(float(rounding[0]), float(rounding[1]))
    draw_shape(vertices, mode=gl.GL_TRIANGLE_FAN, pos=pos, size=size, color=color, alpha=alpha)

@functools.lru_cache(maxsize=10000)
def _setup_rect(rx, ry):
    t = np.linspace(0, np.pi / 2, 1 if max(rx, ry) == 0 else 64)
    s = 1 - np.sin(t); c = 1 - np.cos(t)
    x = [c * rx, 1 - s * rx, 1 - c * rx, s * rx]
    y = [s * ry, c * ry, 1 - s * ry, 1 - c * ry]
    v = np.stack([x, y], axis=-1).reshape(-1, 2)
    return v.astype('float32')

#----------------------------------------------------------------------------

def draw_circle(*, center=0, radius=100, hole=0, color=1, alpha=1):
    hole = np.broadcast_to(np.asarray(hole, dtype='float32'), [])
    vertices = _setup_circle(float(hole))
    draw_shape(vertices, mode=gl.GL_TRIANGLE_STRIP, pos=center, size=radius, color=color, alpha=alpha)

@functools.lru_cache(maxsize=10000)
def _setup_circle(hole):
    t = np.linspace(0, np.pi * 2, 128)
    s = np.sin(t); c = np.cos(t)
    v = np.stack([c, s, c * hole, s * hole], axis=-1).reshape(-1, 2)
    return v.astype('float32')

#----------------------------------------------------------------------------
