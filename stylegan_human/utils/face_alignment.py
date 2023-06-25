# Copyright (c) SenseTime Research. All rights reserved.

import numpy as np
import PIL
import PIL.Image
import scipy
import scipy.ndimage
import dlib
import copy
from PIL import Image

def get_landmark(img, detector, predictor):
    """get landmark with dlib
    :return: np.array shape=(68, 2)
    """
    # detector = dlib.get_frontal_face_detector()
    # dets, _, _ = detector.run(img, 1, -1)
    dets = detector(img, 1)
    for k, d in enumerate(dets):
        shape = predictor(img, d.rect)
    t = list(shape.parts())
    a = []
    for tt in t:
        a.append([tt.x, tt.y])
    lm = np.array(a)
    
    # face rect
    face_rect = [dets[0].rect.left(), dets[0].rect.top(), dets[0].rect.right(), dets[0].rect.bottom()]
    return lm, face_rect


    

def align_face_for_insetgan(img, detector, predictor, output_size=256):
    """
    :param img: numpy array rgb
    :return: PIL Image
    """
    img_cp = copy.deepcopy(img)
    lm, face_rect = get_landmark(img, detector, predictor)

    lm_chin = lm[0: 17]  # left-right
    lm_eyebrow_left = lm[17: 22]  # left-right
    lm_eyebrow_right = lm[22: 27]  # left-right
    lm_nose = lm[27: 31]  # top-down
    lm_nostrils = lm[31: 36]  # top-down
    lm_eye_left = lm[36: 42]  # left-clockwise
    lm_eye_right = lm[42: 48]  # left-clockwise
    lm_mouth_outer = lm[48: 60]  # left-clockwise
    lm_mouth_inner = lm[60: 68]  # left-clockwise

    # Calculate auxiliary vectors.
    eye_left = np.mean(lm_eye_left, axis=0)
    eye_right = np.mean(lm_eye_right, axis=0)
    eye_avg = (eye_left + eye_right) * 0.5
    eye_to_eye = eye_right - eye_left
    mouth_left = lm_mouth_outer[0]
    mouth_right = lm_mouth_outer[6]
    mouth_avg = (mouth_left + mouth_right) * 0.5
    eye_to_mouth = mouth_avg - eye_avg

    # Choose oriented crop rectangle.
    x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
    x /= np.hypot(*x)
    x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
    y = np.flipud(x) * [-1, 1]
    c = eye_avg + eye_to_mouth * 0.1
    quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
    qsize = np.hypot(*x) * 2

    # read image
    # opencv to PIL
    img = PIL.Image.fromarray(img_cp)
    # img = PIL.Image.open(filepath)

    transform_size = output_size
    enable_padding = False

    # Shrink.
    # shrink = int(np.floor(qsize / output_size * 0.5))
    # if shrink > 1:
    #     rsize = (int(np.rint(float(img.size[0]) / shrink)), int(np.rint(float(img.size[1]) / shrink)))
    #     img = img.resize(rsize, PIL.Image.ANTIALIAS)
    #     quad /= shrink
    #     qsize /= shrink

    # Crop.
    border = max(int(np.rint(qsize * 0.1)), 3)
    crop = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))), int(np.ceil(max(quad[:, 0]))),
            int(np.ceil(max(quad[:, 1]))))
    
    # crop = (max(crop[0] - border, 0), max(crop[1] - border, 0), min(crop[2] + border, img.size[0]),
    #         min(crop[3] + border, img.size[1]))
    # img.save("debug/raw.jpg")
    if crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]:
        img = img.crop(crop)
        quad -= crop[0:2]
    # img.save("debug/crop.jpg")
    # Pad.
    # pad = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))), int(np.ceil(max(quad[:, 0]))),
    #        int(np.ceil(max(quad[:, 1]))))
    # pad = (max(-pad[0] + border, 0), max(-pad[1] + border, 0), max(pad[2] - img.size[0] + border, 0),
    #        max(pad[3] - img.size[1] + border, 0))
    # if enable_padding and max(pad) > border - 4:
    #     pad = np.maximum(pad, int(np.rint(qsize * 0.3)))
    #     img = np.pad(np.float32(img), ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), 'reflect')
    #     h, w, _ = img.shape
    #     y, x, _ = np.ogrid[:h, :w, :1]
    #     mask = np.maximum(1.0 - np.minimum(np.float32(x) / pad[0], np.float32(w - 1 - x) / pad[2]),
    #                       1.0 - np.minimum(np.float32(y) / pad[1], np.float32(h - 1 - y) / pad[3]))
    #     blur = qsize * 0.02
    #     img += (scipy.ndimage.gaussian_filter(img, [blur, blur, 0]) - img) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
    #     img += (np.median(img, axis=(0, 1)) - img) * np.clip(mask, 0.0, 1.0)
    #     img = PIL.Image.fromarray(np.uint8(np.clip(np.rint(img), 0, 255)), 'RGB')
    #     quad += pad[:2]

    # Transform.
    # crop shape to transform shape
    # nw = 
    # print(img.size, quad+0.5, np.bound((quad+0.5).flatten()))
    # assert False
    # img = img.transform((transform_size, transform_size), PIL.Image.QUAD, (quad + 0.5).flatten(), PIL.Image.BILINEAR)
    
    # img.save("debug/transform.jpg")
    # if output_size < transform_size:
    img = img.resize((output_size, output_size), PIL.Image.ANTIALIAS)
    # img.save("debug/resize.jpg")
    # print((quad+crop[0:2]).flatten())
    # assert False
    # Return aligned image.
    
    return img, crop, face_rect




def align_face_for_projector(img, detector, predictor, output_size):
    """
    :param filepath: str
    :return: PIL Image
    """

    img_cp = copy.deepcopy(img)
    lm, face_rect = get_landmark(img, detector, predictor)


    lm_chin = lm[0: 17]  # left-right
    lm_eyebrow_left = lm[17: 22]  # left-right
    lm_eyebrow_right = lm[22: 27]  # left-right
    lm_nose = lm[27: 31]  # top-down
    lm_nostrils = lm[31: 36]  # top-down
    lm_eye_left = lm[36: 42]  # left-clockwise
    lm_eye_right = lm[42: 48]  # left-clockwise
    lm_mouth_outer = lm[48: 60]  # left-clockwise
    lm_mouth_inner = lm[60: 68]  # left-clockwise

    # Calculate auxiliary vectors.
    eye_left = np.mean(lm_eye_left, axis=0)
    eye_right = np.mean(lm_eye_right, axis=0)
    eye_avg = (eye_left + eye_right) * 0.5
    eye_to_eye = eye_right - eye_left
    mouth_left = lm_mouth_outer[0]
    mouth_right = lm_mouth_outer[6]
    mouth_avg = (mouth_left + mouth_right) * 0.5
    eye_to_mouth = mouth_avg - eye_avg

    # Choose oriented crop rectangle.
    x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
    x /= np.hypot(*x)
    x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
    y = np.flipud(x) * [-1, 1]
    c = eye_avg + eye_to_mouth * 0.1
    quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
    qsize = np.hypot(*x) * 2

    # read image
    img = PIL.Image.fromarray(img_cp)

    transform_size = output_size
    enable_padding = True

    # Shrink.
    shrink = int(np.floor(qsize / output_size * 0.5))
    if shrink > 1:
        rsize = (int(np.rint(float(img.size[0]) / shrink)), int(np.rint(float(img.size[1]) / shrink)))
        img = img.resize(rsize, PIL.Image.ANTIALIAS)
        quad /= shrink
        qsize /= shrink

    # Crop.
    border = max(int(np.rint(qsize * 0.1)), 3)
    crop = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))), int(np.ceil(max(quad[:, 0]))),
            int(np.ceil(max(quad[:, 1]))))
    crop = (max(crop[0] - border, 0), max(crop[1] - border, 0), min(crop[2] + border, img.size[0]),
            min(crop[3] + border, img.size[1]))
    if crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]:
        img = img.crop(crop)
        quad -= crop[0:2]

    # Pad.
    pad = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))), int(np.ceil(max(quad[:, 0]))),
           int(np.ceil(max(quad[:, 1]))))
    pad = (max(-pad[0] + border, 0), max(-pad[1] + border, 0), max(pad[2] - img.size[0] + border, 0),
           max(pad[3] - img.size[1] + border, 0))
    if enable_padding and max(pad) > border - 4:
        pad = np.maximum(pad, int(np.rint(qsize * 0.3)))
        img = np.pad(np.float32(img), ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), 'reflect')
        h, w, _ = img.shape
        y, x, _ = np.ogrid[:h, :w, :1]
        mask = np.maximum(1.0 - np.minimum(np.float32(x) / pad[0], np.float32(w - 1 - x) / pad[2]),
                          1.0 - np.minimum(np.float32(y) / pad[1], np.float32(h - 1 - y) / pad[3]))
        blur = qsize * 0.02
        img += (scipy.ndimage.gaussian_filter(img, [blur, blur, 0]) - img) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
        img += (np.median(img, axis=(0, 1)) - img) * np.clip(mask, 0.0, 1.0)
        img = PIL.Image.fromarray(np.uint8(np.clip(np.rint(img), 0, 255)), 'RGB')
        quad += pad[:2]

    # Transform.
    img = img.transform((transform_size, transform_size), PIL.Image.QUAD, (quad + 0.5).flatten(), PIL.Image.BILINEAR)
    if output_size < transform_size:
        img = img.resize((output_size, output_size), PIL.Image.ANTIALIAS)

    # Return aligned image.
    return img


def reverse_quad_transform(image, quad_to_map_to, alpha):
    # forward mapping, for simplicity

    result = Image.new("RGBA",image.size)
    result_pixels = result.load()

    width, height = result.size

    for y in range(height):
        for x in range(width):
            result_pixels[x,y] = (0,0,0,0)

    p1 = (quad_to_map_to[0],quad_to_map_to[1])
    p2 = (quad_to_map_to[2],quad_to_map_to[3])
    p3 = (quad_to_map_to[4],quad_to_map_to[5])
    p4 = (quad_to_map_to[6],quad_to_map_to[7])

    p1_p2_vec = (p2[0] - p1[0],p2[1] - p1[1])
    p4_p3_vec = (p3[0] - p4[0],p3[1] - p4[1])

    for y in range(height):
        for x in range(width):
            pixel = image.getpixel((x,y))

            y_percentage = y / float(height)
            x_percentage = x / float(width)

            # interpolate vertically
            pa = (p1[0] + p1_p2_vec[0] * y_percentage, p1[1] + p1_p2_vec[1] * y_percentage) 
            pb = (p4[0] + p4_p3_vec[0] * y_percentage, p4[1] + p4_p3_vec[1] * y_percentage)

            pa_to_pb_vec = (pb[0] - pa[0],pb[1] - pa[1])

            # interpolate horizontally
            p = (pa[0] + pa_to_pb_vec[0] * x_percentage, pa[1] + pa_to_pb_vec[1] * x_percentage)

            try:
                result_pixels[p[0],p[1]] = (pixel[0],pixel[1],pixel[2],min(int(alpha * 255),pixel[3]))
            except Exception:
                pass

    return result