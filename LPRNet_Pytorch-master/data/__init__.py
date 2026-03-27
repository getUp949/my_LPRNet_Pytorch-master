# -*- coding: utf-8 -*-
# 导入 data 包时，将 load_data.py 中所有公开内容（CHARS, CHARS_DICT, LPRDataLoader 等）
# 重新导出为 data 模块的属性，使外部可以通过 data.CHARS、data.LPRDataLoader 的方式访问
# 这是一种"接口暴露"的写法：外部使用 from data import * 时可以直接获取这些符号
from .load_data import *
