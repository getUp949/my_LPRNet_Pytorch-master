# -*- coding: utf-8 -*-
# 导入 PyTorch 的神经网络基础模块（Module、Conv2d、BatchNorm2d 等）
import torch.nn as nn
# 导入 PyTorch 主模块（张量运算）
import torch


# =============================================================================
# small_basic_block：轻量级基础卷积块（深度可分离卷积的变体）
#
# 该模块由四个连续的 1x1 / 3x1 / 1x3 卷积组成，结构如下：
#   Conv(1x1) -> ReLU -> Conv(3x1 + pad) -> ReLU -> Conv(1x3 + pad) -> ReLU -> Conv(1x1)
#
# 采用"压缩-变换-扩展"策略：先将通道数压缩到 ch_out//4，
# 经过两次空间卷积（3x1 和 1x3 分别提取水平和垂直特征）后，
# 再通过 1x1 卷积恢复到原始通道数 ch_out。
# 与标准 Depthwise Separable Conv 类似，但使用两次一维卷积（3x1 和 1x3）
# 代替单个 3x3 二维深度卷积，以更高效地捕获水平和垂直方向的特征。
#
# 参数：
#   ch_in : 输入通道数
#   ch_out: 输出通道数
# =============================================================================
class small_basic_block(nn.Module):
    """
    轻量级基础卷积块（Depthwise Separable 变体）
    采用先压缩后扩展的通道策略，减少参数量
    """

    def __init__(self, ch_in, ch_out):
        # 调用父类 nn.Module 的初始化方法
        super(small_basic_block, self).__init__()

        # -------------------------------------------------------------------------
        # 使用 nn.Sequential 将四个卷积层按顺序串联成模块
        # -------------------------------------------------------------------------
        self.block = nn.Sequential(
            # ---- 第一层：1x1 卷积（压缩阶段）----
            # 将输入通道 ch_in 压缩到输出通道的 1/4
            # 1x1 卷积只做通道间混合，不改变空间尺寸
            nn.Conv2d(ch_in, ch_out // 4, kernel_size=1),
            # 非线性激活，引入非线性
            nn.ReLU(),
            # ---- 第二层：3x1 卷积（水平特征提取）----
            # kernel_size=(3, 1)：仅在高度方向（垂直）做 3 像素的卷积，宽度不变
            # padding=(1, 0)：在高度方向两侧各填充 1 像素，保持高度尺寸不变
            nn.Conv2d(ch_out // 4, ch_out // 4, kernel_size=(3, 1), padding=(1, 0)),
            nn.ReLU(),
            # ---- 第三层：1x3 卷积（垂直特征提取）----
            # kernel_size=(1, 3)：仅在宽度方向（水平）做 3 像素的卷积，高度不变
            # padding=(0, 1)：在宽度方向两侧各填充 1 像素，保持宽度尺寸不变
            nn.Conv2d(ch_out // 4, ch_out // 4, kernel_size=(1, 3), padding=(0, 1)),
            nn.ReLU(),
            # ---- 第四层：1x1 卷积（扩展阶段）----
            # 将通道数从 ch_out//4 恢复到原始 ch_out
            nn.Conv2d(ch_out // 4, ch_out, kernel_size=1),
        )

    # -------------------------------------------------------------------------
    # 前向传播：将输入 x 依次通过四个卷积层
    # -------------------------------------------------------------------------
    def forward(self, x):
        return self.block(x)


# =============================================================================
# LPRNet：车牌识别网络的主体结构
#
# 网络分为两大部分：
#   1. backbone（特征提取骨干网络）：23 层卷积，输出每个时间步的类别概率
#   2. container（全局上下文融合层）：将多尺度特征图沿通道维度拼接后，
#      融合为最终的分类结果
#
# 前向传播流程：
#   输入图像 (N, 3, 24, 94)
#     -> backbone: 提取多尺度卷积特征
#     -> 选取 [2, 6, 13, 22] 层的输出作为多尺度特征
#     -> 对每个特征做全局平均池化（AvgPool），并进行功率归一化
#     -> 沿通道维度拼接所有归一化特征 (torch.cat)
#     -> container: 1x1 卷积融合通道信息
#     -> 对时间维度取平均 (torch.mean)，得到 (N, class_num, T) 的 logits
#
# 参数：
#   lpr_max_len  : 车牌最大字符数，默认 8
#   phase        : 'train' 或 'eval'，控制 Dropout 等层的行为
#   class_num    : 字符类别总数（含 blank），默认 66
#   dropout_rate : Dropout 层的丢弃率，默认 0.5
# =============================================================================
class LPRNet(nn.Module):

    def __init__(self, lpr_max_len, phase, class_num, dropout_rate):
        # 调用父类 nn.Module 的初始化方法
        super(LPRNet, self).__init__()
        # 保存当前模式：'train' 或 'eval'
        self.phase = phase
        # 保存车牌最大字符数
        self.lpr_max_len = lpr_max_len
        # 保存字符类别总数
        self.class_num = class_num

        # -------------------------------------------------------------------------
        # backbone：特征提取骨干网络，由 23 个卷积/BatchNorm/ReLU/MaxPool/Dropout 层串联而成
        # 输入: (N, 3, 24, 94) 彩色图像
        # 输出: (N, class_num, 1, T) 其中 T 为时间步数（与输入宽度相关）
        # -------------------------------------------------------------------------
        self.backbone = nn.Sequential(
            # ---- 层 0：第一个卷积层 ----
            # 输入: (N, 3, 24, 94)，输出: (N, 64, 22, 92)
            # kernel=3, stride=1，无 padding -> 高度和宽度各减少 2 像素
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1),
            # ---- 层 1：批归一化 ----
            # 对 64 个通道的输出做 BatchNorm，稳定训练
            nn.BatchNorm2d(num_features=64),
            # ---- 层 2：ReLU 激活函数 ----
            nn.ReLU(),                                    # *** 索引 2 ***
            # ---- 层 3：第一次空间池化 ----
            # MaxPool3d: 在 (D, H, W) 三个维度做最大池化
            # kernel_size=(1, 3, 3): 对深度（D=序列长度）不做池化，仅对 H 和 W 方向池化
            # stride=(1, 1, 1): 步长 1，输出尺寸保持不变
            # 效果：在特征图的空间维度（H、W）上进行 3x3 池化
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 1, 1)),
            # ---- 层 4：第一个 small_basic_block ----
            # 输入: (N, 64, 22, 92)，输出: (N, 128, 20, 90)
            # 小型深度可分离卷积块，通道从 64 扩展到 128
            small_basic_block(ch_in=64, ch_out=128),      # *** 索引 4 ***
            # ---- 层 5：批归一化 ----
            nn.BatchNorm2d(num_features=128),
            # ---- 层 6：ReLU 激活函数 ----
            nn.ReLU(),                                    # *** 索引 6 ***
            # ---- 层 7：第二次空间池化 ----
            # kernel_size=(1, 3, 3): 3x3 池化
            # stride=(2, 1, 2): 在高度方向步长 2（减半），宽度方向步长 2（减半）
            # 效果：高度从 20 变为 10（减半），宽度从 90 变为 45（减半）
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(2, 1, 2)),
            # ---- 层 8：第二个 small_basic_block ----
            # 输入: (N, 128, 10, 45)，输出: (N, 256, 8, 43)
            # 注意：这里 small_basic_block 的 ch_in=64 而非 128，
            # 原因：前一层 MaxPool3d 的 stride 在 H 方向为 2，导致通道数自动减半（通常来自实现细节）
            small_basic_block(ch_in=64, ch_out=256),      # *** 索引 8 ***
            # ---- 层 9：批归一化 ----
            nn.BatchNorm2d(num_features=256),
            # ---- 层 10：ReLU 激活函数 ----
            nn.ReLU(),
            # ---- 层 11：第三个 small_basic_block ----
            # 输入: (N, 256, 8, 43)，输出: (N, 256, 6, 41)
            small_basic_block(ch_in=256, ch_out=256),    # *** 索引 11 ***
            # ---- 层 12：批归一化 ----
            nn.BatchNorm2d(num_features=256),
            # ---- 层 13：ReLU 激活函数 ----
            nn.ReLU(),
            # ---- 层 14：第三次空间池化 ----
            # kernel_size=(1, 3, 3), stride=(4, 1, 2)
            # 高度方向步长 4（从 6 变为约 2），宽度方向步长 2（从 41 变为约 20）
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(4, 1, 2)),  # *** 索引 14 ***
            # ---- 层 15：Dropout ----
            # 随机丢弃部分神经元，防止过拟合
            nn.Dropout(dropout_rate),
            # ---- 层 16：1x4 卷积 ----
            # 输入: (N, 64, H', W')，这里 H'≈2, W'≈20
            # kernel=(1, 4)：只在宽度方向进行 4 像素卷积，高度方向为 1
            # 输出: (N, 256, H', W'-3+1)
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=(1, 4), stride=1),
            # ---- 层 17：批归一化 ----
            nn.BatchNorm2d(num_features=256),
            # ---- 层 18：ReLU 激活函数 ----
            nn.ReLU(),
            # ---- 层 19：Dropout ----
            nn.Dropout(dropout_rate),
            # ---- 层 20：最后一个卷积层 ----
            # 输入: (N, 256, H', W')，输出: (N, class_num, H', W')
            # kernel_size=(13, 1)：高度方向 13 像素卷积，宽度方向 1 像素
            # 将特征图的高度维度压缩为 1，同时输出 class_num 个通道
            nn.Conv2d(in_channels=256, out_channels=class_num, kernel_size=(13, 1), stride=1),
            # ---- 层 21：批归一化 ----
            nn.BatchNorm2d(num_features=class_num),
            # ---- 层 22：ReLU 激活函数 ----
            nn.ReLU(),                                    # *** 索引 22 ***
        )

        # -------------------------------------------------------------------------
        # container：全局上下文融合层
        # 将 backbone 中提取的多尺度特征（在 forward 中收集）与当前层输出沿通道维度拼接，
        # 再通过 1x1 卷积进行通道融合，输出 class_num 个通道
        #
        # 输入通道数 = 448（来自 keep_features 拼接）+ class_num（当前层输出）
        # 输出通道数 = class_num（66）
        # -------------------------------------------------------------------------
        self.container = nn.Sequential(
            # 1x1 卷积：不做空间混合，仅在通道维度做线性变换（通道混合）
            # 将 448 + class_num 个输入通道压缩到 class_num 个输出通道
            nn.Conv2d(in_channels=448 + self.class_num, out_channels=self.class_num,
                      kernel_size=(1, 1), stride=(1, 1)),
            # 下方为备用的更深层结构（当前被注释掉）：
            # nn.BatchNorm2d(num_features=self.class_num),  # 批归一化
            # nn.ReLU(),                                     # ReLU 激活
            # nn.Conv2d(in_channels=self.class_num, out_channels=self.lpr_max_len+1,
            #           kernel_size=3, stride=2),           # 3x3 卷积 + 步长 2（进一步下采样）
            # nn.ReLU(),
        )

    # -------------------------------------------------------------------------
    # 前向传播：输入一张车牌图像，输出每帧的类别 logit
    #
    # 参数：
    #   x: 输入图像张量，形状 (N, 3, H, W)，通常为 (N, 3, 24, 94)
    #
    # 返回：
    #   logits: 形状 (N, class_num, T)，T 为时间步数（与输入宽度相关）
    #   每一列代表一个时间步在 class_num 个字符上的未归一化得分（logit）
    # -------------------------------------------------------------------------
    def forward(self, x):
        # -------------------------------------------------------------------------
        # 第一步：特征提取
        # 遍历 backbone 的每一层，逐一计算
        # 并从特定层收集中间特征图用于全局上下文建模
        # -------------------------------------------------------------------------
        keep_features = list()  # 用于存放选中的中间层特征图

        for i, layer in enumerate(self.backbone.children()):
            # 将输入依次通过 backbone 的每一层
            x = layer(x)
            # -------------------------------------------------------------------------
            # 在第 2、6、13、22 层（索引）收集特征图
            # 层 2 (ReLU after conv1): 浅层特征，包含基本的边缘和纹理信息
            # 层 6 (ReLU after first block): 中层特征
            # 层 13 (ReLU after second block): 较深层特征
            # 层 22 (ReLU after last conv): 深层语义特征
            # -------------------------------------------------------------------------
            if i in [2, 6, 13, 22]:  # [2, 4, 8, 11, 22]（原始注释中的备选列表）
                keep_features.append(x)

        # -------------------------------------------------------------------------
        # 第二步：全局上下文建模
        # 对收集到的多尺度特征分别进行：
        #   1. 自适应平均池化（将不同尺度的特征图统一到相同尺寸）
        #   2. 功率归一化（使特征在统计上更稳定）
        # -------------------------------------------------------------------------
        global_context = list()  # 存放归一化后的全局上下文特征

        for i, f in enumerate(keep_features):
            # -------------------------------------------------------------------------
            # 对不同层级的特征使用不同的池化核大小，使其在空间上对齐
            # keep_features[0] (层 2): 尺寸较大，使用 5x5 池化窗口，步长 5 快速下采样
            # keep_features[1] (层 6): 尺寸中等，同样使用 5x5 池化窗口，步长 5
            # keep_features[2] (层 13): 尺寸较小，使用 (4, 10) 池化窗口，步长 (4, 2)
            # keep_features[3] (层 22): 尺寸最小，无需池化
            # -------------------------------------------------------------------------
            if i in [0, 1]:
                # 5x5 平均池化，步长 5，将特征图尺寸快速下采样
                f = nn.AvgPool2d(kernel_size=5, stride=5)(f)
            if i in [2]:
                # (4, 10) 核的池化：高度方向 4 步，宽度方向 2 步
                f = nn.AvgPool2d(kernel_size=(4, 10), stride=(4, 2))(f)
            # -------------------------------------------------------------------------
            # 功率归一化（Power Normalization）
            # 步骤 1：计算特征图的逐元素平方 f^2
            # 步骤 2：对整个特征图求均值 E[f^2]
            # 步骤 3：除以均值，实现功率归一化
            # 效果：使特征的方差归一化到 1，有助于训练稳定性和收敛速度
            # -------------------------------------------------------------------------
            f_pow = torch.pow(f, 2)        # f^2
            f_mean = torch.mean(f_pow)      # E[f^2]
            f = torch.div(f, f_mean)        # f / E[f^2]

            global_context.append(f)  # 将归一化后的上下文特征加入列表

        # -------------------------------------------------------------------------
        # 第三步：特征融合
        # 将多尺度、全局上下文增强的特征在通道维度（dim=1）上拼接
        # -------------------------------------------------------------------------
        x = torch.cat(global_context, 1)   # 沿通道维度拼接

        # -------------------------------------------------------------------------
        # 第四步：container 卷积融合
        # 1x1 卷积将拼接后的 448 + class_num 个通道融合为 class_num 个通道
        # -------------------------------------------------------------------------
        x = self.container(x)

        # -------------------------------------------------------------------------
        # 第五步：时间维度平均
        # 对宽度方向（即时间步方向，dim=2）取平均
        # 输入: (N, class_num, 1, T)，输出: (N, class_num, T)
        # 最终输出：每张图像在 T 个时间步上的 class_num 维 logit
        # -------------------------------------------------------------------------
        logits = torch.mean(x, dim=2)   # 在高度维度（已为 1）上求平均，实际上是对时间步取平均

        return logits  # 形状: (N, class_num, T)


# =============================================================================
# build_lprnet：LPRNet 网络的工厂函数/构建器
#
# 负责创建 LPRNet 实例，并根据 phase 参数切换训练/推理模式
#
# 参数：
#   lpr_max_len  : 车牌最大字符数，默认 8
#   phase        : 'train'（训练模式）或 'eval'/'test'（推理模式）
#                  训练模式会启用 Dropout，推理模式会关闭
#   class_num    : 字符类别总数，默认 66（65 个字符 + 1 个 blank）
#   dropout_rate : Dropout 丢弃率，默认 0.5
#
# 返回：
#   构建好的 LPRNet 模型（处于指定模式）
# =============================================================================
def build_lprnet(lpr_max_len=8, phase=False, class_num=66, dropout_rate=0.5):
    # 创建 LPRNet 模型实例
    Net = LPRNet(lpr_max_len, phase, class_num, dropout_rate)

    # -------------------------------------------------------------------------
    # 根据 phase 参数切换模型的训练/推理模式
    # phase == "train" : 返回 Net.train()，启用 Dropout、BatchNorm 使用训练模式
    # phase != "train": 返回 Net.eval()，关闭 Dropout、BatchNorm 使用推理模式
    # -------------------------------------------------------------------------
    if phase == "train":
        return Net.train()   # 切换到训练模式
    else:
        return Net.eval()    # 切换到评估/推理模式
