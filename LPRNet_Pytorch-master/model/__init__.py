# -*- coding: utf-8 -*-
# 导入 model 包时，将 LPRNet.py 中所有公开内容（build_lprnet, LPRNet, small_basic_block 等）
# 重新导出为 model 模块的属性，使外部可以通过 model.build_lprnet 的方式访问
# 这是一种"接口暴露"的写法：外部使用 from model import * 时可以直接获取这些符号
from .LPRNet import *
