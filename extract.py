from glob import glob

from pycocotools.coco import COCO
import os
import shutil
from tqdm import tqdm
import skimage.io as io
import matplotlib.pyplot as plt
import cv2
from PIL import Image, ImageDraw

from prepare import check

# 需要设置的路径
savepath = "COCO_extract"
img_dir = os.path.join(savepath, 'JPEGImages')
anno_dir = os.path.join(savepath, 'Annotations')
datasets_list = ['train2014', 'val2014']

# coco有80类，这里写要提取类的名字，以person为例
classes_names = ['clock']
# 包含所有类别的原coco数据集路径
'''
目录格式如下：
$COCO_PATH
----|annotations
----|train2014
----|val2014
----|test2014
'''
dataDir = 'COCO'

headstr = """\
<annotation>
    <folder>VOC</folder>
    <filename>%s</filename>
    <source>
        <database>My Database</database>
        <annotation>COCO</annotation>
        <image>flickr</image>
        <flickrid>NULL</flickrid>
    </source>
    <owner>
        <flickrid>NULL</flickrid>
        <name>company</name>
    </owner>
    <size>
        <width>%d</width>
        <height>%d</height>
        <depth>%d</depth>
    </size>
    <segmented>0</segmented>
"""
objstr = """\
    <object>
        <name>%s</name>
        <pose>Unspecified</pose>
        <truncated>0</truncated>
        <difficult>0</difficult>
        <bndbox>
            <xmin>%d</xmin>
            <ymin>%d</ymin>
            <xmax>%d</xmax>
            <ymax>%d</ymax>
        </bndbox>
    </object>
"""

tailstr = '''\
</annotation>
'''


# 检查目录是否存在，如果存在，先删除再创建，否则，直接创建
def mkr(path):
    if not os.path.exists(path):
        os.makedirs(path)  # 可以创建多级目录
    # else:
    #     shutil.rmtree(path)
    #     os.makedirs(path)


def id2name(coco):
    classes = dict()
    for cls in coco.dataset['categories']:
        classes[cls['id']] = cls['name']
    return classes


def write_xml(anno_path, head, objs, tail):
    f = open(anno_path, "w")
    f.write(head)
    for obj in objs:
        f.write(objstr % (obj[0], obj[1], obj[2], obj[3], obj[4]))
    f.write(tail)


def save_annotations_and_imgs(dataset, filename, objs):
    # 将图片转为xml，例:COCO_train2017_000000196610.jpg-->COCO_train2017_000000196610.xml
    dst_anno_dir = os.path.join(anno_dir, dataset)
    mkr(dst_anno_dir)
    anno_path = os.path.join(dst_anno_dir, filename[:-3] + 'xml')
    img_path = os.path.join(dataDir, dataset, filename)
    # print("img_path: ", img_path)
    dst_img_dir = os.path.join(img_dir, dataset)
    mkr(dst_img_dir)
    dst_imgpath = os.path.join(dst_img_dir, filename)
    # print("dst_imgpath: ", dst_imgpath)
    img = cv2.imread(img_path)
    # if (img.shape[2] == 1):
    #    print(filename + " not a RGB image")
    #   return
    shutil.copy(img_path, dst_imgpath)

    head = headstr % (filename, img.shape[1], img.shape[0], img.shape[2])
    tail = tailstr
    write_xml(anno_path, head, objs, tail)


# def showimg(coco, dataset, img, classes, cls_id, show=True):
#     global dataDir
#     I = Image.open(os.path.join(dataDir, dataset, img['file_name']))
#     # 通过id，得到注释的信息
#     annIds = coco.getAnnIds(imgIds=img['id'], catIds=cls_id, iscrowd=None)
#     # print(annIds)
#     anns = coco.loadAnns(annIds)
#     # print(anns)
#     # coco.showAnns(anns)
#     objs = []
#     for ann in anns:
#         class_name = classes[ann['category_id']]
#         if class_name in classes_names:
#             # print(class_name)
#             if 'bbox' in ann:
#                 bbox = ann['bbox']
#                 xmin = int(bbox[0])
#                 ymin = int(bbox[1])
#                 xmax = int(bbox[2] + bbox[0])
#                 ymax = int(bbox[3] + bbox[1])
#                 obj = [class_name, xmin, ymin, xmax, ymax]
#                 objs.append(obj)
#                 draw = ImageDraw.Draw(I)
#                 draw.rectangle([xmin, ymin, xmax, ymax])
#     if show:
#         plt.figure()
#         plt.axis('off')
#         plt.imshow(I)
#         plt.show()
#
#     return objs

def extract_classes():
    for dataset in datasets_list:
        annFile = '{}/annotations/instances_{}.json'.format(dataDir, dataset)
        coco = COCO(annFile)
        for cls in classes_names:
            cls_id = coco.getCatIds(catNms=[cls])[0]
            img_ids = coco.getImgIds(catIds=cls_id)
            for imgId in tqdm(img_ids):
                img = coco.loadImgs(imgId)[0]
                filename = img['file_name']
                annIds = coco.getAnnIds(imgIds=img['id'], catIds=cls_id, iscrowd=None)
                anns = coco.loadAnns(annIds)
                objs = []
                for ann in anns:
                    if int(ann['category_id']) == cls_id:
                        if 'bbox' in ann:
                            bbox = ann['bbox']
                            xmin = int(bbox[0])
                            ymin = int(bbox[1])
                            xmax = int(bbox[2]+bbox[0])
                            ymax = int(bbox[3]+bbox[1])
                            obj = [cls, xmin, ymin, xmax, ymax]
                            objs.append(obj)
                save_annotations_and_imgs(dataset, filename, objs)

#
# for dataset in datasets_list:
#     # ./COCO/annotations/instances_train2017.json
#     annFile = '{}/annotations/instances_{}.json'.format(dataDir, dataset)
#
#     # 使用COCO API用来初始化注释数据
#     coco = COCO(annFile)
#
#     id = coco.getCatIds(classes_names)[0]
#     imgIds = coco.getImgIds(catIds=id)
#     print(f'{dataset}有{len(imgIds)}张包含{classes_names[0]}的图片')

    # # 获取COCO数据集中的所有类别
    # classes = id2name(coco)
    # # print(classes)
    # # [1, 2, 3, 4, 6, 8]
    # classes_ids = coco.getCatIds(catNms=classes_names)
    # for cls in classes_names:
    #     # 获取该类的id
    #     cls_id = coco.getCatIds(catNms=[cls])
    #     img_ids = coco.getImgIds(catIds=cls_id)
    #     # print(cls, len(img_ids))
    #     # imgIds=img_ids[0:10]
    #     print(len(img_ids))
    #     for imgId in tqdm(img_ids):
    #         img = coco.loadImgs(imgId)[0]
    #         filename = img['file_name']
    #         # print(filename)
    #         objs = showimg(coco, dataset, img, classes, classes_ids, show=False)
    #         # print(objs)
    #         save_annotations_and_imgs(coco, dataset, filename, objs)


if __name__ == '__main__':
    # extract_classes()
    for dataset in datasets_list:
        if dataset == 'train2014':
            dst_path = 'VOC2007_trainval'
            num = 250
        else:
            dst_path = 'VOC2007_test'
            num = 200
        imgs = glob(os.path.join(img_dir, dataset, '*.jpg'))
        lbs = glob(os.path.join(anno_dir, dataset, '*.xml'))
        dst_img_path = os.path.join("VOCdevkit", dst_path,"JPEGImages")
        dst_lb_path = os.path.join("VOCdevkit", dst_path, "Annotations")
        index = 0
        for img in tqdm(imgs[0:num]):
            index+=1
            filename = os.path.splitext(img.split(os.sep)[-1])[0]
            lb = os.path.join(anno_dir, dataset, filename+'.xml')
            shutil.copy(img, dst_img_path)
            shutil.copy(lb, dst_lb_path)
            if index==num:
                break
        check(dst_path, True)