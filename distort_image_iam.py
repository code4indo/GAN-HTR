# -*- coding: utf-8 -*-
import os
import codecs
import cv2
import numpy as np 
import sys
import shutil
from PIL import Image
import imageio
import matplotlib.pyplot  as plt
import random
from shutil import copyfile
import glob
os.environ["PYTHONIOENCODING"] = "utf-8"
from PIL import Image,ImageFilter
from datetime import datetime
import argparse


f1=codecs.open('logDist_iam.txt','a+','utf-8')

def preprocess2(v, background_dir):
    backgrounds = os.listdir(background_dir)
    ch = random.choice(backgrounds)
 
    from PIL import Image
    img = Image.open(os.path.join(background_dir, ch)).convert('L')
    u = random.randint(1, 4)
    if u == 4:
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
    if u == 1:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    if u == 3:
        img = img.transpose(Image.ROTATE_90)
    if u == 2:
        img = img.transpose(Image.ROTATE_180)
    
    widthb, heightb = img.size

    cv2.imwrite('a.png', v)
    img2 = Image.open('a.png').convert('L')
    
    widthl, heightl = img2.size
    #if widthl>widthb or heightl>heightb:
    #    img=img.resize((widthl+100,heightl+100), Image.ANTIALIAS)
    img.save('output_file.jpg')
    
    img2.save('b.jpg')
    a = plt.imread('b.jpg', 0)
    
    bg = plt.imread('output_file.jpg', 0)
 
    size_a = a.shape[1] * 2 
    size_bg = bg.shape[1]
    
    while size_a > size_bg:
        bg = np.concatenate((bg, bg), axis=1)
        size_bg = size_bg * 2
 
    size_a1 = a.shape[0] * 2
    size_bg1 = bg.shape[0]
    
    while size_a1 > size_bg1:
        bg = np.concatenate((bg, bg), axis=0)
        size_bg1 = size_bg1 * 2
 
    print('ground :', bg.shape)
    print('line: ', a.shape)
    
    p = random.randint(1, 100)
    p2 = random.randint(1, 50)
            
    bg = bg[p:p + a.shape[0], p2:p2 + a.shape[1]]

    param1 = random.randint(3, 7) / 10
    param2 = random.randint(3, 7) / 10
    
    #n=random.randint(10,90)
    a = cv2.addWeighted(bg, param1, a, param2, random.randint(-30, 1))
    return a

def read_file(list_file_path):
    char_file = codecs.open(list_file_path, 'r', 'utf-8')
    lst = []
    for l in char_file:
        lst.append(l.strip())
    return lst

def blur_image_low(img):
    kernel = random.randint(1, 5)
    avging = cv2.blur(img, (kernel, kernel), cv2.BORDER_DEFAULT) 
    return avging

def blur_image_hight(img):
    kernel = random.randint(6, 15)
    avging = cv2.blur(img, (kernel, kernel), cv2.BORDER_DEFAULT) 
    return avging

def list_files(dir):
    r = []
    for root, dirs, files in os.walk(dir):
        for name in files:
            r.append(os.path.join(root, name))
    return r
 
def erodecv(img):
    # Taking a matrix of size 5 as the kernel 
    k = random.randint(2, 4)
    kernel = np.ones((k, k), np.uint8) 
    # The first parameter is the original image, 
    # kernel is the matrix with which image is  
    # convolved and third parameter is the number  
    # of iterations, which will determine how much  
    # you want to erode/dilate a given image.  
    img_erosion = cv2.erode(img, kernel, iterations=1) 
    return img_erosion

def dilatecv(img):
    # Taking a matrix of size 5 as the kernel 
    k = random.randint(2, 3)
    kernel = np.ones((k, k), np.uint8) 
    # The first parameter is the original image, 
    # kernel is the matrix with which image is  
    # convolved and third parameter is the number  
    # of iterations, which will determine how much  
    # you want to erode/dilate a given image.  
    img_erosion = cv2.dilate(img, kernel, iterations=1)
    return img_erosion

def distort_line(image):
    # Compute histogram
    im = image
    im = 255 - im

    thik1 = random.randint(2, 8)
    thik2 = random.randint(2, 5)
    thik3 = random.randint(5, 10)
    thik4 = random.randint(2, 10)
    newimage = image.copy()
    index_of_highest_peak = random.randint(20, 40)
    ind1 = index_of_highest_peak
    ind2 = index_of_highest_peak + random.randint(40, 50)
    ind3 = index_of_highest_peak - random.randint(40, 50)
    image_widh = im.shape[1] 
 
    i1 = random.randint(10, image_widh - 5)
    i2 = random.randint(40, image_widh - 20)
    i3 = random.randint(10, image_widh - 30)
    i4 = random.randint(5, image_widh - 10)

    cv2.line(newimage, pt1=(i1, 0), pt2=(i1, 400), color=(0, 0, 0), thickness=thik1)
    cv2.line(newimage, pt1=(i3, 0), pt2=(i3, 400), color=(0, 0, 0), thickness=thik2)
    cv2.line(newimage, pt1=(i2, 0), pt2=(i2, 400), color=(0, 0, 0), thickness=thik3)
    cv2.line(newimage, pt1=(i4, 0), pt2=(i4, 400), color=(0, 0, 0), thickness=thik4)
    
    return newimage

def distortion(set_type, background_dir, num_images=None):
    i = 0
    
    # Define paths based on the set_type (train, valid, or test)
    base_gt_path = 'datasets/iam_raw/'
    # Adjust the path based on the set type, e.g., 'train', 'validation', 'test'
    set_folder = 'validation' if set_type == 'valid' else set_type
    gt = os.path.join(base_gt_path, set_folder, 'images/')
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ww = os.path.join('datasets/iam_distorted', timestamp)
    list_file = f'Sets/list_{set_type}_iam.txt'

    if not os.path.exists(gt):
        print(f"ERROR: Ground truth directory not found: {gt}")
        return
    if not os.path.exists(ww):
        os.makedirs(ww)
        print(f"Created output directory: {ww}")
    if not os.path.isfile(list_file):
        print(f"ERROR: List file not found: {list_file}")
        return

    listf = read_file(list_file)
    if num_images is not None and num_images > 0:
        listf = listf[:num_images]
        
    #random.shuffle(listf)
    nbfiles = len(listf)
    sp1 = nbfiles / 8
    sp2 = 2 * nbfiles / 8
    sp3 = 3 * nbfiles / 8
    sp4 = 4 * nbfiles / 8
    sp5 = 5 * nbfiles / 8
    sp6 = 6 * nbfiles / 8
    sp7 = 7 * nbfiles / 8
    sp8 = 8 * nbfiles / 8

    print(f"Processing {len(listf)} files from {list_file}")
    for filename in listf:
        print(f"Processing file {i+1}/{nbfiles}: {filename}")

        # Construct the full path for the input image, assuming .png format
        image_path = os.path.join(gt, filename + '.png')
        if not os.path.exists(image_path):
            print(f"  - Warning: Image file not found, skipping: {image_path}")
            continue

        a = plt.imread(image_path)
        
        plt.imsave('imagex.jpg', a, cmap='gray')
        a = plt.imread('imagex.jpg')
        im1 = a
        im2 = a
        im3 = a
        
        output_path = os.path.join(ww, filename + '.png')

        #########add background
        if i < sp1: 
            # im1=distort_line(im1)
            # im1=dilatecv(im1)
            # im1=blur_image_low(im1)
            # im1=preprocess2(im1)
            # imageio.imwrite(output_path,preprocess2(im1))
            # f1.write(filename + ' dilate,blur low,2 preprocess'+ '\n')
            ##add blur
            im2 = dilatecv(im2)
            bluredl = blur_image_hight(im2)
            f1.write(filename + ' dilate,blur highest,2 preprocess' + '\n')
            imageio.imwrite(output_path, preprocess2(bluredl, background_dir))
        # ########add low blur and background
        elif i >= sp1 and i < sp2:
            ##add blur
            im2 = dilatecv(im2)
            bluredl = blur_image_hight(im2)
            f1.write(filename + ' dilate,blur highest,2 preprocess' + '\n')
            imageio.imwrite(output_path, preprocess2(bluredl, background_dir))
        elif i >= sp1 and i < sp2:
            ##add blur
            im3 = dilatecv(im3)
            bluredh = blur_image_hight(im3)
            bluredh = distort_line(bluredh)
            f1.write(filename + ' dilate,blur highest,line, preprocess' + '\n')
            imageio.imwrite(output_path, preprocess2(bluredh, background_dir))
        elif i >= sp2 and i < sp3:
            x = dilatecv(im1)
            f1.write(filename + ' dilate, preprocess' + '\n')
            imageio.imwrite(output_path, preprocess2(x, background_dir))
        elif i >= sp3 and i < sp4:
            x = preprocess2(im1, background_dir)
            x = dilatecv(x)
            f1.write(filename + ' dilate preprocess' + '\n'),
            imageio.imwrite(output_path, preprocess2(x, background_dir))
        elif i >= sp4 and i < sp5:
            x = dilatecv(im1)
            f1.write(filename + ' dilate,blur highest,preprocess' + '\n')
            bluredl = blur_image_hight(x)
            imageio.imwrite(output_path, preprocess2(bluredl, background_dir))
        elif i >= sp5 and i < sp6:
            x = erodecv(im1)
            gauss = distort_line(x)
            bluredl = blur_image_low(gauss)
            f1.write(filename + ' erode,blur low,line, preprocess' + '\n')
            imageio.imwrite(output_path, preprocess2(bluredl, background_dir))
        elif i >= sp6 and i < sp7:
            bluredl = erodecv(im1)
            
            f1.write(filename + ' erode, preprocess' + '\n')
            imageio.imwrite(output_path, preprocess2(bluredl, background_dir))

        
        else:
            ##add blur
            im2 = dilatecv(im2)
            bluredl = blur_image_hight(im2)
            f1.write(filename + ' dilate,blur highest,2 preprocess' + '\n')
            imageio.imwrite(output_path, preprocess2(bluredl, background_dir))
        
        i = i + 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Distort IAM dataset images.')
    parser.add_argument('set_type', choices=['train', 'valid', 'test'], 
                        help="The set to process: 'train', 'valid', or 'test'")
    parser.add_argument('--background_dir', type=str, default='datasets/background_images/',
                        help='Path to the directory containing background images.')
    parser.add_argument('--num_images', type=int, default=None,
                        help='Number of images to process from the set.')
    
    args = parser.parse_args()

    print(f"--- Starting distortion process for IAM '{args.set_type}' set ---")
    distortion(args.set_type, args.background_dir, args.num_images)
    print(f"--- Distortion process for '{args.set_type}' set finished ---")