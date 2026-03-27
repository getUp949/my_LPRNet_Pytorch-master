# -*- coding: utf-8 -*-
# 导入 PyTorch 的 Dataset 相关类，用于自定义数据集
from torch.utils.data import *
# 导入 imutils 库的 paths 模块，用于遍历指定目录下的图像文件
from imutils import paths
# 导入 numpy，用于数值计算和数组操作
import numpy as np
# 导入 random，用于打乱图像路径列表的顺序
import random
# 导入 OpenCV，用于图像的读取、缩放和预处理
import cv2
# 导入 os，用于文件路径操作
import os

# =============================================================================
# 定义车牌字符集：共 65 个类别（0~64）
# 第 0~30 位：34 个中国省级行政区简称（直辖市、省、自治区）
# 第 31~40 位：10 个阿拉伯数字 0~9
# 第 41~64 位：24 个英文字母（A~Z，但去掉易与数字混淆的 I 和 O）
# 第 65 位：'-'（分隔符，在 CTC 解码时作为 blank 符号使用）
# =============================================================================
CHARS = ['京', '沪', '津', '渝', '冀', '晋', '蒙', '辽', '吉', '黑',
         '苏', '浙', '皖', '闽', '赣', '鲁', '豫', '鄂', '湘', '粤',
         '桂', '琼', '川', '贵', '云', '藏', '陕', '甘', '青', '宁',
         '新',                          # 第 0~30 位，共 31 个省份简称
         '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',  # 第 31~40 位，10 个数字
         'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K',  # 第 41~50 位，部分字母
         'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',  # 第 51~60 位，部分字母
         'W', 'X', 'Y', 'Z', 'I', 'O', '-'                  # 第 61~64 位，剩余字母及分隔符
         ]

# =============================================================================
# 将字符映射为索引的字典，方便后续通过字符查索引
# 示例：CHARS_DICT['A'] -> 41
# =============================================================================
CHARS_DICT = {char: i for i, char in enumerate(CHARS)}


# =============================================================================
# LPRDataLoader：车牌图像数据集类，继承自 torch.utils.data.Dataset
# 负责从磁盘加载车牌图像、预处理并返回（图像, 标签, 标签长度）
# =============================================================================
class LPRDataLoader(Dataset):
    """
    参数：
        img_dir    : 图像所在的文件夹路径（支持多文件夹逗号分隔）
        imgSize    : 目标图像尺寸 [宽, 高]，默认 [94, 24]
        lpr_max_len: 车牌最大字符数，默认 8
        PreprocFun : 图像预处理函数，默认为 None（使用内置的 transform）
    """

    def __init__(self, img_dir, imgSize, lpr_max_len, PreprocFun=None):
        # 保存传入的文件夹路径列表（img_dir 可以是字符串或字符串列表）
        self.img_dir = img_dir
        # 用于存放所有图像文件的完整路径
        self.img_paths = []

        # 遍历每个文件夹，将其中的图像路径收集到 img_paths 中
        for i in range(len(img_dir)):
            # imutils.paths.list_images 会递归遍历 img_dir[i] 目录下的所有图片文件
            self.img_paths += [el for el in paths.list_images(img_dir[i])]

        # 打乱图像路径顺序，实现数据增强的随机性（训练时尤为重要）
        random.shuffle(self.img_paths)

        # 保存目标图像尺寸
        self.img_size = imgSize
        # 保存车牌最大字符数
        self.lpr_max_len = lpr_max_len

        # 如果传入了自定义预处理函数，则使用它；否则使用内置的 transform 方法
        if PreprocFun is not None:
            self.PreprocFun = PreprocFun
        else:
            self.PreprocFun = self.transform

    # -------------------------------------------------------------------------
    # 返回数据集的样本总数，供 DataLoader 使用以确定批次数量
    # -------------------------------------------------------------------------
    def __len__(self):
        return len(self.img_paths)

    # -------------------------------------------------------------------------
    # 根据索引获取单个样本：对图像进行预处理，从文件名中解析标签
    # 返回：(预处理后的图像 numpy 数组, 标签列表, 标签长度)
    # -------------------------------------------------------------------------
    def __getitem__(self, index):
        # 获取第 index 张图像的完整路径
        filename = self.img_paths[index]
        # 规范化路径（处理不同操作系统的路径分隔符差异）
        filename = os.path.normpath(filename)

        # -------------------------------------------------------------------------
        # 读取图像（OpenCV 方式，BGR 格式）
        # 注意：OpenCV 在 Windows 下对含中文路径支持较差，因此做 fallback 处理
        # -------------------------------------------------------------------------
        try:
            Image = cv2.imdecode(np.fromfile(filename, dtype=np.uint8), cv2.IMREAD_COLOR)
        except Exception:
            raise FileNotFoundError(f"Cannot read image: {filename}")

        if Image is None:
            raise FileNotFoundError(f"Cannot read image: {filename}")

        # 二次检查读取结果，确保图像有效
        if Image is None:
            raise FileNotFoundError(f"Cannot read image: {filename}")

        # -------------------------------------------------------------------------
        # 图像尺寸检查与缩放：将图像 resize 到目标尺寸 [宽, 高]
        # 默认目标尺寸为 [94, 24]（宽 94 像素，高 24 像素）
        # -------------------------------------------------------------------------
        height, width, _ = Image.shape
        if height != self.img_size[1] or width != self.img_size[0]:
            # cv2.resize 接受的参数格式为 (宽, 高)
            Image = cv2.resize(Image, self.img_size)

        # 对图像进行归一化和通道转换预处理
        Image = self.PreprocFun(Image)

        # -------------------------------------------------------------------------
        # 从文件名中解析车牌标签
        # 文件名格式示例："京A12345_0.jpg" 或 "冀D88888.jpg"
        # -------------------------------------------------------------------------
        basename = os.path.basename(filename)          # 取文件名部分，去掉目录路径
        imgname, suffix = os.path.splitext(basename)   # 分离文件名和扩展名

        # 车牌号通常在文件名最前面，用 '-' 或 '_' 分隔；取第一段即为车牌号
        imgname = imgname.split("-")[0].split("_")[0]

        #TODO
        #print("标签:", imgname)

        # -------------------------------------------------------------------------
        # 将车牌号的每个字符转换为对应的类别索引
        # 示例："京A12345" -> [0, 41, 31, 32, 33, 34, 35]
        # -------------------------------------------------------------------------
        label = list()
        for c in imgname:
            # one_hot_base = np.zeros(len(CHARS))
            # one_hot_base[CHARS_DICT[c]] = 1
            label.append(CHARS_DICT[c])

        # -------------------------------------------------------------------------
        # 长度校验：标准民用车牌长度为 8
        # 若长度为 8（新能源车牌），调用 check() 验证格式合法性
        # 新能源车牌规则：第 3 位（索引 2）必须是 D 或 F；末位必须是 D 或 F
        # -------------------------------------------------------------------------
        if len(label) == 8:
            if self.check(label) == False:
                print(imgname)  # 打印出错的标签
                assert 0, "Error label ^~^!!!"  # 抛出异常终止程序

        # -------------------------------------------------------------------------
        # 返回预处理后的图像、标签列表（字符索引）、标签长度
        # 标签长度用于 CTC Loss 计算
        # -------------------------------------------------------------------------
        return Image, label, len(label)

    # -------------------------------------------------------------------------
    # 内置的图像预处理方法：将图像归一化到 [-1, 1]，并将通道顺序从 HWC 转为 CHW
    # -------------------------------------------------------------------------
    def transform(self, img):
        # 将图像数据类型转换为 float32（浮点运算精度要求）
        img = img.astype('float32')
        # 第一步归一化：将像素值从 [0, 255] 中心化到 [-127.5, 127.5]
        img -= 127.5
        # 第二步归一化：乘以 0.0078125（即 1/128），将值域映射到 [-1, 1]
        img *= 0.0078125
        # OpenCV 读取的图像是 HWC（高×宽×通道）格式，BGR 通道顺序
        # PyTorch 要求 CHW（通道×高×宽）格式，转换为 (2, 0, 1) 即 (C, H, W)
        img = np.transpose(img, (2, 0, 1))

        return img

    # -------------------------------------------------------------------------
    # check()：校验新能源车牌（长度为 8）的格式合法性
    # 新能源车牌的特点：第 3 位（省份缩写后）是字母 D 或 F，末位也是字母 D 或 F
    # -------------------------------------------------------------------------
    def check(self, label):
        # label[2] 是车牌第 3 位（省份缩写后的第一位）
        # label[-1] 是车牌的最后一个字符
        # 两者中至少有一个必须是字母 D（电动）或 F（非电动），否则为非法格式
        if label[2] != CHARS_DICT['D'] and label[2] != CHARS_DICT['F'] \
                and label[-1] != CHARS_DICT['D'] and label[-1] != CHARS_DICT['F']:
            print("Error label, Please check!")  # 打印错误标签提示
            return False  # 返回 False 表示校验失败
        else:
            return True   # 校验通过
