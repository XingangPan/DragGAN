# Copyright (c) SenseTime Research. All rights reserved.


import os
import argparse 
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from utils.ImagesDataset import ImagesDataset

import cv2
import time
import copy
import imutils

# for openpose body keypoint detector : # (src:https://github.com/Hzzone/pytorch-openpose)
from openpose.src import util
from openpose.src.body import Body

# for paddlepaddle human segmentation : #(src: https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.5/contrib/PP-HumanSeg/)
from PP_HumanSeg.deploy.infer import Predictor as PP_HumenSeg_Predictor

import math
def angle_between_points(p0,p1,p2):
    if p0[1]==-1 or p1[1]==-1 or p2[1]==-1:
        return -1
    a = (p1[0]-p0[0])**2 + (p1[1]-p0[1])**2
    b = (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2
    c = (p2[0]-p0[0])**2 + (p2[1]-p0[1])**2 
    if a * b == 0:
        return -1
    return math.acos((a+b-c) / math.sqrt(4*a*b)) * 180 / math.pi


def crop_img_with_padding(img, keypoints, rect):
    person_xmin,person_xmax, ymin, ymax= rect
    img_h,img_w,_ = img.shape    ## find body center using keypoints
    middle_shoulder_x = keypoints[1][0]
    middle_hip_x = (keypoints[8][0] + keypoints[11][0]) // 2
    mid_x = (middle_hip_x + middle_shoulder_x) // 2    
    mid_y = (ymin + ymax) // 2
    ## find which side (l or r) is further than center x, use the further side
    if abs(mid_x-person_xmin) > abs(person_xmax-mid_x): #left further
        xmin = person_xmin
        xmax = mid_x + (mid_x-person_xmin)
    else:
        ############### may be negtive
        ### in this case, the script won't output any image, leave the case like this
        ### since we don't want to pad human body
        xmin = mid_x - (person_xmax-mid_x)   
        xmax = person_xmax 

    w = xmax - xmin
    h = ymax - ymin
    ## pad rectangle to w:h = 1:2 ## calculate desired border length
    if h / w >= 2: #pad horizontally
        target_w = h // 2
        xmin_prime = int(mid_x - target_w / 2)
        xmax_prime = int(mid_x + target_w / 2)
        if xmin_prime < 0:
            pad_left = abs(xmin_prime)# - xmin
            xmin = 0
        else:
            pad_left = 0
            xmin = xmin_prime
        if xmax_prime > img_w:
            pad_right = xmax_prime - img_w
            xmax = img_w
        else:
            pad_right = 0
            xmax = xmax_prime

        cropped_img = img[int(ymin):int(ymax), int(xmin):int(xmax)]
        im_pad = cv2.copyMakeBorder(cropped_img, 0, 0, int(pad_left),  int(pad_right), cv2.BORDER_REPLICATE) 
    else: #pad vertically
        target_h = w * 2
        ymin_prime = mid_y - (target_h / 2)
        ymax_prime = mid_y + (target_h / 2) 
        if ymin_prime < 0: 
            pad_up = abs(ymin_prime)# - ymin
            ymin = 0
        else:
            pad_up = 0
            ymin = ymin_prime
        if ymax_prime > img_h:
            pad_down = ymax_prime - img_h
            ymax = img_h
        else:
            pad_down = 0
            ymax = ymax_prime
        print(ymin,ymax, xmin,xmax, img.shape)

        cropped_img = img[int(ymin):int(ymax), int(xmin):int(xmax)]
        im_pad = cv2.copyMakeBorder(cropped_img, int(pad_up), int(pad_down), 0,
                                    0, cv2.BORDER_REPLICATE) 
    result = cv2.resize(im_pad,(512,1024),interpolation = cv2.INTER_AREA)
    return result


def run(args):
    os.makedirs(args.output_folder, exist_ok=True)
    dataset = ImagesDataset(args.image_folder, transforms.Compose([transforms.ToTensor()]))
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    body_estimation = Body('openpose/model/body_pose_model.pth')

    total = len(dataloader)
    print('Num of dataloader : ', total)
    os.makedirs(f'{args.output_folder}', exist_ok=True)
    # os.makedirs(f'{args.output_folder}/middle_result', exist_ok=True)
    
    ## initialzide HumenSeg
    human_seg_args = {}
    human_seg_args['cfg'] = 'PP_HumanSeg/export_model/deeplabv3p_resnet50_os8_humanseg_512x512_100k_with_softmax/deploy.yaml'
    human_seg_args['input_shape'] = [1024,512]
    human_seg_args['save_dir'] = args.output_folder
    human_seg_args['soft_predict'] = False
    human_seg_args['use_gpu'] = True
    human_seg_args['test_speed'] = False
    human_seg_args['use_optic_flow'] = False
    human_seg_args['add_argmax'] = True
    human_seg_args= argparse.Namespace(**human_seg_args)
    human_seg = PP_HumenSeg_Predictor(human_seg_args)

    from tqdm import tqdm
    for fname, image in tqdm(dataloader):
        # try:
        ## tensor to numpy image
        fname = fname[0]
        print(f'Processing \'{fname}\'.')
        
        image = (image.permute(0, 2, 3, 1) * 255).clamp(0, 255)
        image = image.squeeze(0).numpy() # --> tensor to numpy, (H,W,C)
        # avoid super high res img
        if image.shape[0] >= 2000: # height  ### for shein image
            ratio = image.shape[0]/1200 #height
            dim = (int(image.shape[1]/ratio),1200)#(width, height)
            image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        ## create segmentation
        # mybg = cv2.imread('mybg.png') 
        comb, segmentation, bg, ori_img = human_seg.run(image,None)  #mybg) 
        # cv2.imwrite('comb.png',comb)  # [0,255]
        # cv2.imwrite('alpha.png',segmentation*255) # segmentation [0,1] --> [0.255]
        # cv2.imwrite('bg.png',bg)  #[0,255]
        # cv2.imwrite('ori_img.png',ori_img) # [0,255]

        masks_np = (segmentation* 255)# .byte().cpu().numpy() #1024,512,1
        mask0_np = masks_np[:,:,0].astype(np.uint8)#[0, :, :]
        contours = cv2.findContours(mask0_np,  cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(contours)
        c = max(cnts, key=cv2.contourArea)
        extTop = tuple(c[c[:, :, 1].argmin()][0])
        extBot = tuple(c[c[:, :, 1].argmax()][0])
        extBot = list(extBot)
        extTop = list(extTop)
        pad_range = int((extBot[1]-extTop[1])*0.05)
        if (int(extTop[1])<=5 and int(extTop[1])>0) and (comb.shape[0]>int(extBot[1]) and int(extBot[1])>=comb.shape[0]-5): #seg mask already reaches to the edge
            #pad with pure white, top 100 px, bottom 100 px
            comb= cv2.copyMakeBorder(comb,pad_range+5,pad_range+5,0,0,cv2.BORDER_CONSTANT,value=[255,255,255]) 
        elif int(extTop[1])<=0 or int(extBot[1])>=comb.shape[0]:
            print('PAD: body out of boundary', fname) #should not happened
            return {}
        else:
            comb = cv2.copyMakeBorder(comb, pad_range+5, pad_range+5, 0, 0, cv2.BORDER_REPLICATE) #105 instead of 100: give some extra space
        extBot[1] = extBot[1] + pad_range+5
        extTop[1] = extTop[1] + pad_range+5

        extLeft = tuple(c[c[:, :, 0].argmin()][0])
        extRight = tuple(c[c[:, :, 0].argmax()][0])
        extLeft = list(extLeft)
        extRight = list(extRight)
        person_ymin = int(extTop[1])-pad_range # 100
        person_ymax = int(extBot[1])+pad_range # 100 #height
        if person_ymin<0 or person_ymax>comb.shape[0]: # out of range
            return {}
        person_xmin = int(extLeft[0])
        person_xmax = int(extRight[0])
        rect =  [person_xmin,person_xmax,person_ymin, person_ymax]
        # recimg = copy.deepcopy(comb)
        # cv2.rectangle(recimg,(person_xmin,person_ymin),(person_xmax,person_ymax),(0,255,0),2)
        # cv2.imwrite(f'{args.output_folder}/middle_result/{fname}_rec.png',recimg)

        ## detect keypoints
        keypoints, subset = body_estimation(comb)
        # print(keypoints, subset, len(subset))
        if len(subset) != 1 or (len(subset)==1 and subset[0][-1]<15): 
            print(f'Processing \'{fname}\'. Please import image contains one person only. Also can check segmentation mask. ')
            continue

        # canvas = copy.deepcopy(comb)
        # canvas = util.draw_bodypose(canvas, keypoints, subset, show_number=True)
        # cv2.imwrite(f'{args.output_folder}/middle_result/{fname}_keypoints.png',canvas)

        comb = crop_img_with_padding(comb, keypoints, rect)

        
        cv2.imwrite(f'{args.output_folder}/{fname}.png', comb)
        print(f' -- Finished processing \'{fname}\'. --')
        # except:
        #     print(f'Processing \'{fname}\'. Not satisfied the alignment strategy.')
        
        
if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    
    t1 = time.time()
    arg_formatter = argparse.ArgumentDefaultsHelpFormatter
    description = 'StyleGAN-Human data process'
    parser = argparse.ArgumentParser(formatter_class=arg_formatter,
                                     description=description)
    parser.add_argument('--image-folder', type=str, dest='image_folder')
    parser.add_argument('--output-folder', dest='output_folder', default='results', type=str)
    # parser.add_argument('--cfg', dest='cfg for segmentation', default='PP_HumanSeg/export_model/ppseg_lite_portrait_398x224_with_softmax/deploy.yaml', type=str)

    print('parsing arguments')
    cmd_args = parser.parse_args()
    run(cmd_args)

    print('total time elapsed: ', str(time.time() - t1))