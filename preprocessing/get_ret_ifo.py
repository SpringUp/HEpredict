import numpy as np
import cv2
import os
from preprocessing.slide_window import slide_window
from utils.common import logger


def get_ret_ifo(xy_list: list, slide, svs_address,
                image_dir_split, window_size: int, stride: int,
                points_con_thre: int, area_ratio_thre: float):
    """From the xy_list ,getting the information which can help get a min circumscribed rectangle
    :param xy_list: 点的坐标列表，坐标以列表的形式表示
    :param slide:读取的svs文件
    :param image_dir_split 存储分割后的图片路径
    :param window_size:窗口大小
    :param stride:窗口步长
    :param points_con_thre: 轮廓内点的个数阈值
    :param area_ratio_thre: 面积阈值
    """
    (filepath, filename) = os.path.split(svs_address)#svs_address得到文件路径及文件名
    tiles_dir = '-'.join(filename.split('-')[:3])#取文件名的前三块（病人代号）
    if not os.path.exists(os.path.join(image_dir_split, tiles_dir)):#储存分割后图片路径+病人代号
        os.mkdir(os.path.join(image_dir_split, tiles_dir))#上路径不存在则创建文件夹
    image_address = image_dir_split + '/' + tiles_dir + '/' + filename#image_address=储存分割后图片路径+病人代号+svs的文件名

    for i in range(len(xy_list)):
        if len(xy_list[i]) == 0:
            continue

        points=xy_list[i]

        logger.info("Dealing with the {0}th Cancer area of {1}".format(i, svs_address.split('/')[-1]))


        contours = np.array(points)#轮廓
        x, y, w, h = cv2.boundingRect(contours)#x，y是矩阵左上点的坐标，w，h是外接矩阵的宽和高


        try:
            slide_window(slide, image_address, x, y, w, h,
                             window_size, stride,
                               contours,
                             points_con_thre, area_ratio_thre, svs_address)#调用slide_window得到小窗口并进行去噪等处理
        except Exception as e:
                logger.warn(e)


