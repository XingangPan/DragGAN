# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import contextlib
import imgui

#----------------------------------------------------------------------------

def set_default_style(color_scheme='dark', spacing=9, indent=23, scrollbar=27):
    s = imgui.get_style()
    s.window_padding        = [spacing, spacing]
    s.item_spacing          = [spacing, spacing]
    s.item_inner_spacing    = [spacing, spacing]
    s.columns_min_spacing   = spacing
    s.indent_spacing        = indent
    s.scrollbar_size        = scrollbar
    s.frame_padding         = [4, 3]
    s.window_border_size    = 1
    s.child_border_size     = 1
    s.popup_border_size     = 1
    s.frame_border_size     = 1
    s.window_rounding       = 0
    s.child_rounding        = 0
    s.popup_rounding        = 3
    s.frame_rounding        = 3
    s.scrollbar_rounding    = 3
    s.grab_rounding         = 3

    getattr(imgui, f'style_colors_{color_scheme}')(s)
    c0 = s.colors[imgui.COLOR_MENUBAR_BACKGROUND]
    c1 = s.colors[imgui.COLOR_FRAME_BACKGROUND]
    s.colors[imgui.COLOR_POPUP_BACKGROUND] = [x * 0.7 + y * 0.3 for x, y in zip(c0, c1)][:3] + [1]

#----------------------------------------------------------------------------

@contextlib.contextmanager
def grayed_out(cond=True):
    if cond:
        s = imgui.get_style()
        text = s.colors[imgui.COLOR_TEXT_DISABLED]
        grab = s.colors[imgui.COLOR_SCROLLBAR_GRAB]
        back = s.colors[imgui.COLOR_MENUBAR_BACKGROUND]
        imgui.push_style_color(imgui.COLOR_TEXT, *text)
        imgui.push_style_color(imgui.COLOR_CHECK_MARK, *grab)
        imgui.push_style_color(imgui.COLOR_SLIDER_GRAB, *grab)
        imgui.push_style_color(imgui.COLOR_SLIDER_GRAB_ACTIVE, *grab)
        imgui.push_style_color(imgui.COLOR_FRAME_BACKGROUND, *back)
        imgui.push_style_color(imgui.COLOR_FRAME_BACKGROUND_HOVERED, *back)
        imgui.push_style_color(imgui.COLOR_FRAME_BACKGROUND_ACTIVE, *back)
        imgui.push_style_color(imgui.COLOR_BUTTON, *back)
        imgui.push_style_color(imgui.COLOR_BUTTON_HOVERED, *back)
        imgui.push_style_color(imgui.COLOR_BUTTON_ACTIVE, *back)
        imgui.push_style_color(imgui.COLOR_HEADER, *back)
        imgui.push_style_color(imgui.COLOR_HEADER_HOVERED, *back)
        imgui.push_style_color(imgui.COLOR_HEADER_ACTIVE, *back)
        imgui.push_style_color(imgui.COLOR_POPUP_BACKGROUND, *back)
        yield
        imgui.pop_style_color(14)
    else:
        yield

#----------------------------------------------------------------------------

@contextlib.contextmanager
def item_width(width=None):
    if width is not None:
        imgui.push_item_width(width)
        yield
        imgui.pop_item_width()
    else:
        yield

#----------------------------------------------------------------------------

def scoped_by_object_id(method):
    def decorator(self, *args, **kwargs):
        imgui.push_id(str(id(self)))
        res = method(self, *args, **kwargs)
        imgui.pop_id()
        return res
    return decorator

#----------------------------------------------------------------------------

def button(label, width=0, enabled=True):
    with grayed_out(not enabled):
        clicked = imgui.button(label, width=width)
    clicked = clicked and enabled
    return clicked

#----------------------------------------------------------------------------

def collapsing_header(text, visible=None, flags=0, default=False, enabled=True, show=True):
    expanded = False
    if show:
        if default:
            flags |= imgui.TREE_NODE_DEFAULT_OPEN
        if not enabled:
            flags |= imgui.TREE_NODE_LEAF
        with grayed_out(not enabled):
            expanded, visible = imgui.collapsing_header(text, visible=visible, flags=flags)
        expanded = expanded and enabled
    return expanded, visible

#----------------------------------------------------------------------------

def popup_button(label, width=0, enabled=True):
    if button(label, width, enabled):
        imgui.open_popup(label)
    opened = imgui.begin_popup(label)
    return opened

#----------------------------------------------------------------------------

def input_text(label, value, buffer_length, flags, width=None, help_text=''):
    old_value = value
    color = list(imgui.get_style().colors[imgui.COLOR_TEXT])
    if value == '':
        color[-1] *= 0.5
    with item_width(width):
        imgui.push_style_color(imgui.COLOR_TEXT, *color)
        value = value if value != '' else help_text
        changed, value = imgui.input_text(label, value, buffer_length, flags)
        value = value if value != help_text else ''
        imgui.pop_style_color(1)
    if not flags & imgui.INPUT_TEXT_ENTER_RETURNS_TRUE:
        changed = (value != old_value)
    return changed, value

#----------------------------------------------------------------------------

def drag_previous_control(enabled=True):
    dragging = False
    dx = 0
    dy = 0
    if imgui.begin_drag_drop_source(imgui.DRAG_DROP_SOURCE_NO_PREVIEW_TOOLTIP):
        if enabled:
            dragging = True
            dx, dy = imgui.get_mouse_drag_delta()
            imgui.reset_mouse_drag_delta()
        imgui.end_drag_drop_source()
    return dragging, dx, dy

#----------------------------------------------------------------------------

def drag_button(label, width=0, enabled=True):
    clicked = button(label, width=width, enabled=enabled)
    dragging, dx, dy = drag_previous_control(enabled=enabled)
    return clicked, dragging, dx, dy

#----------------------------------------------------------------------------

def drag_hidden_window(label, x, y, width, height, enabled=True):
    imgui.push_style_color(imgui.COLOR_WINDOW_BACKGROUND, 0, 0, 0, 0)
    imgui.push_style_color(imgui.COLOR_BORDER, 0, 0, 0, 0)
    imgui.set_next_window_position(x, y)
    imgui.set_next_window_size(width, height)
    imgui.begin(label, closable=False, flags=(imgui.WINDOW_NO_TITLE_BAR | imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_MOVE))
    dragging, dx, dy = drag_previous_control(enabled=enabled)
    imgui.end()
    imgui.pop_style_color(2)
    return dragging, dx, dy

#----------------------------------------------------------------------------

def click_hidden_window(label, x, y, width, height, img_w, img_h, enabled=True):
    imgui.push_style_color(imgui.COLOR_WINDOW_BACKGROUND, 0, 0, 0, 0)
    imgui.push_style_color(imgui.COLOR_BORDER, 0, 0, 0, 0)
    imgui.set_next_window_position(x, y)
    imgui.set_next_window_size(width, height)
    imgui.begin(label, closable=False, flags=(imgui.WINDOW_NO_TITLE_BAR | imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_MOVE))
    clicked, down = False, False
    img_x, img_y = 0, 0
    if imgui.is_mouse_down():
        posx, posy = imgui.get_mouse_pos()
        if posx >= x and posx < x + width and posy >= y and posy < y + height:
            if imgui.is_mouse_clicked():
                clicked = True
            down = True
            img_x = round((posx - x) / (width - 1) * (img_w - 1))
            img_y = round((posy - y) / (height - 1) * (img_h - 1))
    imgui.end()
    imgui.pop_style_color(2)
    return clicked, down, img_x, img_y

#----------------------------------------------------------------------------
