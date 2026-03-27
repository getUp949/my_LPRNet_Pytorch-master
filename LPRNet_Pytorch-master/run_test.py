import torch
import cv2
import numpy as np
from model.LPRNet import build_lprnet

# 1. 配置基本信息
device = torch.device("cpu")  # 先用CPU跑，最稳
img_path = "plate_0_0.jpg"  # 你的测试图片名字
model_path = "weights/Final_LPRNet_model.pth"  # 你的权重路径

# 2. 准备字符映射表（中国车牌标准的字符库）
CHARS = ['京', '沪', '津', '渝', '冀', '晋', '蒙', '辽', '吉', '黑',
         '苏', '浙', '皖', '闽', '赣', '鲁', '豫', '鄂', '湘', '粤',
         '桂', '琼', '川', '贵', '云', '藏', '陕', '甘', '青', '宁',
         '新', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
         'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M',
         'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
         'I', 'O', '']  # 这里的字符顺序必须与你下载的模型训练时一致

# 3. 加载模型
lprnet = build_lprnet(lpr_max_len=8, phase=False, class_num=len(CHARS), dropout_rate=0).to(device)
lprnet.load_state_dict(torch.load(model_path, map_location=device))
lprnet.eval()
print("模型加载成功！")

# 4. 图像预处理 (LPRNet的标准输入是 94x24)
img = cv2.imread(img_path)
img = cv2.resize(img, (94, 24))
img = img.astype('float32')
img -= 127.5
img *= 0.0078125
img = np.transpose(img, (2, 0, 1))
img = torch.from_numpy(img).unsqueeze(0).to(device)

# 5. 推理
with torch.no_grad():
    preds = lprnet(img)
    preds = preds.cpu().detach().numpy()  # (1, 68, 18)

# 6. 解析结果 (使用最简单的贪心解码)
preb_labels = list()
for i in range(preds.shape[0]):
    pred = preds[i, :, :]
    pred_label = list()
    for j in range(pred.shape[1]):
        pred_label.append(np.argmax(pred[:, j], axis=0))

    # 去重和去除空白符
    no_repeat_blank_label = list()
    pre_c = pred_label[0]
    if pre_c != len(CHARS) - 1:
        no_repeat_blank_label.append(pre_c)
    for c in pred_label[1:]:
        if (pre_c == c) or (c == len(CHARS) - 1):
            if c == len(CHARS) - 1:
                pre_c = c
            continue
        no_repeat_blank_label.append(c)
        pre_c = c
    preb_labels.append(no_repeat_blank_label)

# 7. 打印结果
plate_num = ""
for i in preb_labels[0]:
    plate_num += CHARS[i]
print(f"识别结果为: {plate_num}")