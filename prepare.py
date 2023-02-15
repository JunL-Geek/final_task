import os
import shutil
import random

import numpy as np
import pandas as pd
from tqdm import tqdm
from glob import glob
import matplotlib.pyplot as plt

from collections import defaultdict
import xml.etree.ElementTree as ET


def check(year='VOC2007_trainval', show=False):
    """
    输入数据文件名，返回有图没标注文件和有标注文件没图的数据路径
    """
    ######################################################################################################
    ##########################本节代码检查只有图或只有标注文件的情况##########################################
    #######################################################################################################
    data_path = os.path.join("VOCdevkit", year)
    imgs_path = os.path.join(data_path, 'JPEGImages')
    anns_path = os.path.join(data_path, 'Annotations')
    # 获取图片文件
    img_names = set([os.path.splitext(i)[0] for i in os.listdir(imgs_path)])
    ann_names = set([os.path.splitext(i)[0] for i in os.listdir(anns_path)])
    print(
        "########################################################################################数据集{}检验结果如下：######################################################################################################".format(
            year))
    if not len(img_names):
        print('    该数据集没有图片')
        return
    img_ann = img_names - ann_names  # 有图没标注文件
    ann_img = ann_names - img_names  # 有标注文件没有图

    if len(img_ann):
        print("        有图片没标注文件的图片是：{} 等（只列前50个） 注意检查这些图片是否是背景图片".format(
            {v for k, v in enumerate(img_ann) if k < 50}))

    else:
        print("        所有图片都有对应标注文件")
    if len(ann_img):
        print("        有标注文件没有图片的标注文件是：{}(只列前50个）".format(
            {v for k, v in enumerate(ann_img) if k < 50}))

    else:
        print("        所有标注文件都有对应图片")

    #####################################################################################################
    #######本节代码对于上节检查结果有问题的图片和标注文件统一移动到结果文件夹中进行下一步查看 ##################
    #####################################################################################################

    result_path = os.path.join(data_path, year + '_result')
    if os.path.exists(result_path):
        print('        结果文件{}已经存在，请检查'.format(result_path))
    else:
        os.makedirs(result_path)
    if len(ann_img) + len(img_ann):
        # 把只有图或只有标注文件的数据集全部移出来
        if (not os.path.exists(result_path)):
            os.makedirs(result_path)
        else:
            print('             存在有图无标注或有标注无图的文件，另结果文件{}已经存在，请检查'.format(result_path))

            # return
        img_anns = [os.path.join(imgs_path, i + '.jpg') for i in img_ann]
        ann_imgs = [os.path.join(anns_path, i + '.xml') for i in ann_img]
        if len(img_anns):
            for img in img_anns:
                shutil.move(img, result_path)
            print('                 移动只有图无标注文件完成')
        if len(ann_img):
            for ann in ann_imgs:
                shutil.move(ann, result_path)
            print('                 移动只有标注文件无图完成')
    ###################################################################################################
    ##########本节内容提取分类文件夹标注文件夹中所有的分类类别，这个部分由于数据可能是#######################
    ##########多个人标的，所在对于使用数据的人还是要看一下分类的，很有必要           #######################

    ann_names_new = [os.path.join(anns_path, i) for i in os.listdir(anns_path)]  # 得新获取经过检查处理的标注文件
    total_images_num = len(ann_names_new)
    classes = list()  # 用来存放所有的标注框的分类名称
    img_boxes = list()  # 用来存放单张图片的框的个数
    hw_percents = list()  # 用来存放图像的高宽比，因为图像是要进行resize的，所以可能会有resize和scaled resize区分
    num_imgs = defaultdict(int)  # 存放每个分类有多少张图片出现
    num_boxes = dict()  # 存放每个分类有多少个框出现
    h_imgs = list()  # 存放每张图的高
    w_imgs = list()  # 存放每张图的宽
    area_imgs = list()  # 存放每张图的面积
    h_boxes = defaultdict(list)  # 存放每个分类框的高
    w_boxes = defaultdict(list)  # 存放每个分类框的宽
    area_boxes = defaultdict(list)  # 存放每个分类框的面积
    area_percents = defaultdict(list)  # 存放每个分类框与图像面积的百分比
    for ann in tqdm(ann_names_new):
        try:
            in_file = open(ann)
            tree = ET.parse(in_file)
        except:
            print(f"打开标注文件{ann}失败,文件将被处理")
            shutil.move(ann, result_path)
            im_path = os.path.join(ann.split(os.sep)[0], ann.split(os.sep)[1], 'JPEGImages',
                                   os.path.splitext(ann)[0].split(os.sep)[-1] + '.jpg')
            shutil.move(im_path, result_path)
            continue

        root = tree.getroot()
        try:
            size = root.find('size')
            # print image_id
            w = int(size.find('width').text)
            h = int(size.find('height').text)
        except:
            print(f"取标注尺寸错误，标注文件{ann}将被处理")
            shutil.move(ann, result_path)
            im_path = os.path.join(ann.split(os.sep)[0], ann.split(os.sep)[1], 'JPEGImages',
                                   os.path.splitext(ann)[0].split(os.sep)[-1] + '.jpg')
            shutil.move(im_path, result_path)
            continue

        img_area = w * h
        if img_area < 100:
            print(f"有标注文件{ann}无图片尺寸，将被处理")
            shutil.move(ann, result_path)
            im_path = os.path.join(ann.split(os.sep)[0], ann.split(os.sep)[1], 'JPEGImages',
                                   os.path.splitext(ann)[0].split(os.sep)[-1] + '.jpg')
            shutil.move(im_path, result_path)
            continue

        img_boxes.append(len(root.findall('object')))
        if not len(root.findall('object')):
            print(f"有标注文件{ann}但没有标注物体，将被处理")
            shutil.move(ann, result_path)
            i_path = os.path.join(ann.split(os.sep)[0], ann.split(os.sep)[1], 'JPEGImages',
                                  os.path.splitext(ann)[0].split(os.sep)[-1] + '.jpg')
            shutil.move(i_path, result_path)
            continue
        img_classes = []
        ok_flag = True
        for obj in root.iter('object'):
            difficult = obj.find('difficult').text
            cls_name = obj.find('name').text
            if isinstance(cls_name, type(None)):
                print(f"标注框类名有问题，标注文件将被处理，类名:{cls_name},标注文件：{ann}")
                shutil.move(ann, result_path)
                ok_flag = False
                continue
            elif isinstance(cls_name, str) and len(cls_name) < 2:
                ok_flag = False
                print(f"标注框类名有问题，标注文件将被处理，类名:{cls_name},标注文件：{ann}")
                shutil.move(ann, result_path)
                continue
            else:
                pass

            # if  int(difficult) == 1:
            #     continue
            xmlbox = obj.find('bndbox')
            b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
                 float(xmlbox.find('ymax').text))  # 左，右，上，下

            if int(b[1] - b[0]) == 0 or int(b[3] - b[2]) == 0:
                ok_flag = False
                print(f"有零存在,框为点或直线，将被处理，边框：{b},标注文件：{ann},类名称：{cls_name}，标注文件：{ann}")
                shutil.move(ann, result_path)

            box_area = (b[1] - b[0]) * (b[3] - b[2])
            area_percent = round(np.sqrt(box_area / float(img_area)), 3) * 100
            hw_percents.append(float(h / w))
            if not (cls_name in classes):
                classes.append(cls_name)
            img_classes.append(cls_name)
            num_boxes[cls_name] = num_boxes.get(cls_name, 0) + 1
            h_boxes[cls_name].append(int(b[3] - b[2]))
            w_boxes[cls_name].append(int(b[1] - b[0]))
            area_boxes[cls_name].append(int(box_area))
            area_percents[cls_name].append(area_percent)
        if ok_flag:
            h_imgs.append(h)
            w_imgs.append(w)
            area_imgs.append(img_area)
            for img_cls_name in set(img_classes):
                num_imgs[img_cls_name] = num_imgs.get(img_cls_name, 0) + 1

    classes = sorted(classes)
    print(
        f"数据集{year}一共有{total_images_num}张合格的标注图片,{sum(img_boxes)}个标注框，平均每张图有{round(sum(img_boxes) / total_images_num, 2)}个标注框；一共有{len(classes)}个分类，分别是{classes}；图片中标注框个数最少是{min(img_boxes)}, \
    最多是{max(img_boxes)}.图片高度最小值是{min(h_imgs)},最大值是{max(h_imgs)};图片宽度最小值是{min(w_imgs)},最大值是{max(w_imgs)}; \
    图片面积最小值是{min(area_imgs)},最大值是{max(area_imgs)} ;图片高宽比最小值是{round(min(hw_percents), 2)}，图片高宽比最大值是{round(max(hw_percents), 2)}")
    num_imgs_class = [num_imgs[class_name] for class_name in classes]
    num_boxes_class = [num_boxes[class_name] for class_name in classes]  # 各分类的标注框个数
    min_h_boxes = [min(h_boxes[class_name]) for class_name in classes]  # 各分类标注框高度最小值
    max_h_boxes = [max(h_boxes[class_name]) for class_name in classes]  # 各分类标注框高度最大值
    min_w_boxes = [min(w_boxes[class_name]) for class_name in classes]  # 各分类标注框宽度最小值
    max_w_boxes = [max(w_boxes[class_name]) for class_name in classes]  # 各分类标注框宽度最大值
    min_area_boxes = [min(area_boxes[class_name]) for class_name in classes]  # 各分类标注框面积最小值
    max_area_boxes = [max(area_boxes[class_name]) for class_name in classes]  # 各分类标注框面积最大值
    min_area_percents = [min(area_percents[class_name]) for class_name in classes]  # 各分类标注框面积与图像面积比最小值
    max_area_percents = [max(area_percents[class_name]) for class_name in classes]  # 各分类标注框面积与图像面积比最大值
    result = {'cls_names': classes, 'images': num_imgs_class, 'objects': num_boxes_class, 'min_h_bbox': min_h_boxes,
              'max_h_bbox': max_h_boxes, 'min_w_bbox': min_w_boxes,
              'max_w_bbox': max_w_boxes, 'min_area_bbox': min_area_boxes, 'max_area_bbox': max_area_boxes,
              'min_area_box/img': min_area_percents, 'max_area_box/img': max_area_percents}
    # 显示所有列(参数设置为None代表显示所有行，也可以自行设置数字)
    pd.set_option('display.max_columns', None)
    # 显示所有行
    pd.set_option('display.max_rows', None)
    # 设置数据的显示长度，默认为50
    pd.set_option('max_colwidth', 50)
    # 禁止自动换行(设置为Flase不自动换行，True反之)
    pd.set_option('expand_frame_repr', False)
    result_df = pd.DataFrame(result)
    print(result_df)
    # plt.figure(figsize=(10.8,6.4))
    # result_df.iloc[:,1:3].plot(kind='bar',)
    if show:
        ##############################################画各个类别图片数与框数的直方图############################################################
        plt.figure(figsize=(15, 6.4))

        x1 = [i + 4 * i for i in range(len(classes))]
        x2 = [i + 2 for i in x1]
        y1 = [int(num_boxes[cl]) for cl in classes]
        y2 = [int(num_imgs[cl]) for cl in classes]
        lb1 = ["" for i in x1]
        lb2 = classes
        plt.bar(x1, y1, alpha=0.7, width=2, color='b', label='objects', tick_label=lb1)
        plt.bar(x2, y2, alpha=0.7, width=2, color='r', label='images', tick_label=lb2)
        plt.xticks(rotation=45)
        # plt.axis('off')
        plt.legend()

        # plt.savefig
        ##############################################画单张图标注框数量的直方图################################################################
        # 接着用直方图把这些结果画出来

        plt.figure(figsize=(15, 6.4))

        # 定义组数，默认60
        # 定义一个间隔大小
        a = 1

        # 得出组数
        group_num = int((max(img_boxes) - min(img_boxes)) / a)

        n, bins, patches = plt.hist(x=img_boxes, bins=group_num, color='c', edgecolor='red', density=False, rwidth=0.8)
        for k in range(len(n)):
            plt.text(bins[k], n[k] * 1.02, int(n[k]), fontsize=12,
                     horizontalalignment="center")  # 打标签，在合适的位置标注每个直方图上面样本数
        # 组距
        distance = int((max(img_boxes) - min(img_boxes)) / group_num)
        if distance < 1:
            distance = 1

        plt.xticks(range(min(img_boxes), max(img_boxes) + 2, distance), fontsize=8)
        # 辅助显示设置

        plt.xlabel('number of bbox in each image')
        plt.ylabel('image numbers')
        plt.xticks(rotation=45)
        plt.title(
            f"The number of bbox min:{round(np.min(img_boxes), 2)},max:{round(np.max(img_boxes), 2)} \n mean:{round(np.mean(img_boxes), 2)} std:{round(np.std(img_boxes), 2)}")
        plt.grid(True)
        plt.tight_layout()
        ##############################################画单张图高宽比的直方图################################################################
        plt.figure(figsize=(15, 6.4))
        # 定义组数，默认60
        a = 0.1

        # 得出组数
        group_num = int((max(hw_percents) - min(hw_percents)) / a)

        n, bins, patches = plt.hist(x=hw_percents, bins=group_num, color='c', edgecolor='red', density=False,
                                    rwidth=0.8)
        for k in range(len(n)):
            plt.text(bins[k], n[k] * 1.02, int(n[k]), fontsize=12,
                     horizontalalignment="center")  # 打标签，在合适的位置标注每个直方图上面样本数
        # 组距
        distance = int((max(hw_percents) - min(hw_percents)) / group_num)

        if distance < 1:
            distance = 1
        plt.xticks(range(int(min(hw_percents)), int(max(hw_percents)) + 2, distance), fontsize=8)
        # 辅助显示设置
        plt.xlabel('image height/width in each image')
        plt.ylabel('image numbers')
        plt.xticks(rotation=45)
        plt.title(
            f"image height/width min:{round(np.min(hw_percents))},max:{round(np.max(hw_percents), 2)} \n mean:{round(np.mean(hw_percents), 2)} std:{round(np.std(hw_percents), 2)}")
        plt.grid(True)
        plt.tight_layout()
        ##############################################画各个分类框图面积比直方图################################################################
        plt.figure(figsize=(8 * 3, 8 * round(len(classes) / 3)))
        for i, name in enumerate(classes):
            plt.subplot(int(np.ceil(len(classes) / 3)), 3, i + 1)
            # 定义组数，默认60
            a = 5

            # 得出组数
            group_num = int((max(area_percents[name]) - min(area_percents[name])) / a)
            n, bins, patches = plt.hist(x=area_percents[name], bins=group_num, color='c', edgecolor='red',
                                        density=False, rwidth=0.8)
            for k in range(len(n)):
                plt.text(bins[k], n[k] * 1.02, int(n[k]), fontsize=12,
                         horizontalalignment="center")  # 打标签，在合适的位置标注每个直方图上面样本数
            # 组距
            distance = int((max(area_percents[name]) - min(area_percents[name])) / group_num)

            if distance < 1:
                distance = 1
            plt.xticks(range(int(min(area_percents[name])), int(max(area_percents[name])) + 2, distance), fontsize=8)
            # 辅助显示设置
            plt.xlabel('area percent bbox/img')
            plt.ylabel('boxes numbers')
            plt.xticks(rotation=45)
            plt.title(
                f"id {i + 1} class {name} area percent min:{round(np.min(area_percents[name]), 2)},max:{round(np.max(area_percents[name]), 2)} \n mean:{round(np.mean(area_percents[name]), 2)} std:{round(np.std(area_percents[name]), 2)}")
            plt.grid(True)
            plt.tight_layout()

def remove_classes(year='VOC2007_trainval', classes=None):
    """
    输入数据文件名，将指定分类数据移除
    :param year: 数据文件名
    :param classes: 列表指定的类
    :return:
    """
    data_path = os.path.join('VOCdevkit', year)
    imgs_path = os.path.join(data_path, 'JPEGImages')
    anns_path = os.path.join(data_path, 'Annotations')

    if not len(os.listdir(imgs_path)):
        print('    该数据集没有图片')
        return

    result_path = os.path.join(data_path, year+'_result')
    if os.path.exists(result_path):
        print('    结果文件{}已经存在，请检查'.format(result_path))
    else:
        os.makedirs(result_path)

    if classes is not None:
        source_anns = os.listdir(anns_path)
        for source_ann in tqdm(source_anns):
            tree = ET.parse(os.path.join(anns_path, source_ann))
            root = tree.getroot()
            result = root.findall("object")
            for obj in result:
                if obj.find('name').text in classes:
                    shutil.move(os.path.join(anns_path, source_ann), result_path)
                    img_path = os.path.join(data_path, 'JPEGImages', os.path.splitext(source_ann)[0])+'.jpg'
                    shutil.move(img_path, result_path)
                    break
    else:
        pass

def convert(size, box):
    """
    将绑定盒转换为（x,y,w,h)格式
    :param size: 图片的宽和高
    :param box: 绑定盒的左右上下
    :return:
    """
    dw = 1 / size[0]
    dh = 1 / size[1]
    w = box[1]-box[0]
    h = box[3]-box[2]
    x = (box[1] + box[0])/2
    y = (box[3] + box[2])/2
    w = w*dw
    h = h*dh
    x = x*dw
    y = y*dh
    return (x,y,w,h)

def convert2yolo(year = 'VOC2007_trainval', dstpath=None):
    """
    将图片及标注转换为yolo格式
    :param srcpath: 源路径
    :param dstpath: 目的路径
    :return:
    """
    if os.path.exists(dstpath):
        shutil.rmtree(dstpath)
        os.makedirs(dstpath)
    dstimgs = os.path.join(dstpath, 'images')
    dstlbs = os.path.join(dstpath, 'labels')
    os.makedirs(dstimgs)
    os.makedirs(dstlbs)

    srcpath = os.path.join('VOCdevkit', year)
    imgs = glob(os.path.join(srcpath, 'JPEGImages', '*.jpg'))
    lbs = glob(os.path.join(srcpath, 'Annotations', '*.xml'))

    for img in tqdm(imgs):
        shutil.copy(img, dstimgs)

    classes = []
    for lb in lbs:
        tree = ET.parse(lb)
        root = tree.getroot()
        objs = root.findall('object')
        for obj in objs:
            classname = obj.find('name').text
            classes.append(classname)
    classes = sorted(list(set(classes)))
    classnames = [ j+'\n' if i<len(classes)-1 else j for i,j in enumerate(classes)]
    with open(os.path.join(dstpath, 'classes.txt'), 'w') as f:
        f.writelines(classnames)
    for lb in tqdm(lbs):
        name=os.path.splitext(os.path.split(lb)[-1])[0]
        out_file = os.path.join(dstlbs, name+'.txt')
        fout = open(out_file, 'w')
        tree = ET.parse(lb)
        root = tree.getroot()
        size = root.find('size')
        width = int(size.find('width').text)
        height = int(size.find('height').text)
        objs = root.findall('object')
        for obj in objs:
            difficult = obj.find('difficult').text
            if int(difficult)==1:
                continue
            classname = obj.find('name').text
            classindex = classes.index(classname)
            bbox = obj.find('bndbox')
            b = (float(bbox.find('xmin').text), float(bbox.find('xmax').text), float(bbox.find('ymin').text), float(bbox.find('ymax').text))
            bb = convert((width, height), b)
            fout.write(str(classindex) + " " + " ".join([str(a)for a in bb]) + '\n')
        fout.close()

def partition_tra_val(yolo_data_path = None, tra_ratio=0.7):
    abs_path = os.getcwd()
    print(abs_path)
    imgs_path = os.path.join(yolo_data_path, 'trainval', 'images')
    lines = []
    with open(os.path.join(yolo_data_path, 'trainval.txt'), 'w') as f:
        for img in tqdm(os.listdir(imgs_path)):
            line = os.path.join(abs_path, imgs_path, img)
            lines.append(line)
        f.writelines([j+'\n' if i<len(lines)-1 else j for i, j in enumerate(lines)])
        random.shuffle(lines)
        train_lines = lines[0:int(tra_ratio*len(lines))]
        val_lines = lines[int(tra_ratio * len(lines)):]
        with open(os.path.join(yolo_data_path, 'train.txt'), 'w') as f:
            f.writelines([j + '\n' if i < len(train_lines) - 1 else j for i, j in enumerate(train_lines)])
        with open(os.path.join(yolo_data_path, 'val.txt'), 'w') as f:
            f.writelines([j + '\n' if i < len(val_lines) - 1 else j for i, j in enumerate(val_lines)])


if __name__=='__main__':
    # remove_classes('VOC2007_trainval', ['bus'])
    # check('VOC2007_trainval', True)
    # remove_classes('VOC2007_test', ['bus'])
    # check('VOC2007_test', True)
    partition_tra_val('YOLO_data', tra_ratio=0.8)

