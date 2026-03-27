
import cv2
import os
os.environ["PYTHONIOENCODING"] = "utf-8"

# ===== 字符映射 =====
provinces = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑",
             "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤",
             "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新"]

alphabets = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K',
             'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
             'W', 'X', 'Y', 'Z']

ads = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K',
       'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
       'W', 'X', 'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']


# ===== 解析车牌号 =====
def decode_plate(plate_str):
    plate_list = plate_str.split('_')

    plate = provinces[int(plate_list[0])]
    plate += alphabets[int(plate_list[1])]

    for i in plate_list[2:]:
        plate += ads[int(i)]

    return plate

def imwrite_unicode(path, img):
    ext = os.path.splitext(path)[1]
    result, buf = cv2.imencode(ext, img)
    if result:
        buf.tofile(path)

# ===== 主函数 =====
def process_ccpd(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if not filename.endswith(".jpg"):
            continue

        try:
            # 文件名分割
            parts = filename.split('-')

            # ===== bbox =====
            bbox = parts[2]
            left_up, right_down = bbox.split('_')
            x1, y1 = map(int, left_up.split('&'))
            x2, y2 = map(int, right_down.split('&'))

            # ===== 车牌号 =====
            plate_str = parts[4]
            plate = decode_plate(plate_str)

            # ===== 读取图片 =====
            img_path = os.path.join(input_dir, filename)
            img = cv2.imread(img_path)

            if img is None:
                continue

            # ===== 裁剪 =====
            plate_img = img[y1:y2, x1:x2]

            print(repr(plate))
            # ===== 保存 =====
            save_path = os.path.join(output_dir, f"{plate}.jpg")
            imwrite_unicode(save_path, plate_img)

        except Exception as e:
            print(f"跳过: {filename}, 错误: {e}")


# ===== 运行 =====
if __name__ == "__main__":
    input_dir = "data/ccpd_base"  # 你的CCPD图片目录
    output_dir = "data/train"  # 输出目录

    process_ccpd(input_dir, output_dir)