import argparse
import openslide
import os
import sys
from glob import glob
from os.path import join
from preprocessing.load_xml import load_xml
from preprocessing.get_ret_ifo import get_ret_ifo
from utils.common import logger
import multiprocessing

path_wd = os.path.dirname(sys.argv[0])
#sys.argv[0]文件名；得到当前文件的绝对路径
sys.path.append(path_wd)
if not path_wd == '':
    os.chdir(path_wd)#改变当前工作目录到指定路径
need_save = False

def multiprocessing_segmentation(xml, index, images_dir_split, size_square, prepare_type):
    xy_list = load_xml(xml[index])#读取坐标
    if os.path.exists(xml[index].split("xml")[0]+prepare_type):#xml前的路径+prepare_type（svs)
        image_address = xml[index].split("xml")[0] + prepare_type#上述路径存在付给image_address
        slide = openslide.open_slide(image_address)#读取图片
        # image_large = \
        get_ret_ifo(xy_list, slide, image_address, images_dir_split,
                    size_square, size_square, 3, 0.3)#调用get_ret_ifo小窗口的切割筛选


def prepare_data(images_dir_root, images_dir_split, size_square, prepare_type):
    num_name = 0

    image_dir_list = glob(join(images_dir_root, r'*/'))#image_dir_list=images_dir_root(图像路径下所有目录）
    logger.info(join(images_dir_root, r'*/'))
    segmentation_pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())#开多进程
    for image_dir in image_dir_list:#图像
        xml_files = glob(join(image_dir, '*.xml'))#xml_files=所有 图像路径+xml 文件
        if len(xml_files) == 0:
            # raise FileNotFoundError
            continue
        for index_xml in range(len(xml_files)):#range返回可迭代对象，返回文件长度的数字（1，2，3，4，5……）
            num_name += 1
            logger.info("xml_files: {}".format(xml_files[index_xml]))#计数，xml文件个数（应该是指一张图片的）

            segmentation_pool.apply(multiprocessing_segmentation,
                                    (xml_files, index_xml, images_dir_split, size_square, prepare_type))

    logger.info('tiles are done.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='svs to tiles')#创建解析器
    parser.add_argument('--slide_image_root', type=str, default="/mnt/hgfs/cescimage/imagesplus")
    parser.add_argument('--tiles_image_root', type=str, default="/mnt/hgfs/cescimage/tiles")#切割好后图片路径
    parser.add_argument('--size_square', type=int, default=512)
    parser.add_argument('--prepare_types', type=str, default="svs")
    args = parser.parse_args()

    logger.info('Processing svs images to tiles')
    available_policies = ["svs", "ndpi"]
    assert args.prepare_types in available_policies, "svs or ndpi slide support only"
    prepare_data(args.slide_image_root, args.tiles_image_root, args.size_square, args.prepare_types)
