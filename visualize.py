import os
from glob import glob
import shutil

import cv2
from tqdm import tqdm

color_list = [(255, 0, 0), (156, 102, 31), (255, 0, 255), (0, 255, 0), (0, 255, 255),
              (64, 224, 205), (8, 46, 84), (34, 139, 34), (0, 0, 255), (25, 25, 112),
              (255, 255, 0), (255, 153, 18), (227, 207, 87), (85, 102, 0), (128, 42, 42),
              (188, 143, 143), (160, 32, 240), (218, 112, 214), (118, 128, 105), (255, 192, 203)]

def yolo2voc(yolo_bbox, size):
    x, y, w, h = yolo_bbox
    width, height = size
    x_c = x*width
    y_c = y*height
    w = w*width
    h = h*height
    x1 = x_c - w/2
    y1 = y_c - h/2
    x2 = x1+w
    y2 = y1+h
    return (x1, y1, x2, y2)

def draw_objects(imgs_path, lbs_path, save_path, classes, num, extra=True):
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    os.makedirs(save_path)
    if extra == False:
        str = '*.jpg'
    else:
        str = 'COCO*.jpg'
    imgs = glob(os.path.join(imgs_path, str))
    imgs = imgs[0:num]
    for img in tqdm(imgs):
        lb = os.path.splitext(img.split(os.path.sep)[-1])[0]+'.txt'
        lb = os.path.join(lbs_path, lb)
        image = cv2.imread(img)
        height, width, depth = image.shape
        with open(lb, 'r') as f:
            lines = f.readlines()
        for line in lines:
            line = line.strip('\n')
            bbox = line.split()
            index = int(bbox[0])
            x1, y1, x2, y2 = yolo2voc(map(float,bbox[1:]), (width, height))
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color=color_list[index], thickness=2)
            class_name = classes[index]
            cv2.putText(image, class_name, (int(x1), int(y1)-3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_list[index], thickness=1)
        # cv2.imshow('image', image)
        # cv2.waitKey()
        cv2.imwrite(os.path.join(save_path, img.split(os.path.sep)[-1]), image)




if __name__ == '__main__':
    imgs_path = os.path.join('YOLO_trainval','images')
    lbs_path = os.path.join('YOLO_trainval', 'labels')
    save_path = os.path.join('YOLO_trainval', 'visual')
    classes = []
    with open(os.path.join('YOLO_trainval', 'classes.txt'), 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip('\n')
            classes.append(line)
    print(classes)
    draw_objects(imgs_path=imgs_path, lbs_path=lbs_path, save_path=save_path, classes=classes, num=-1, extra=False)
