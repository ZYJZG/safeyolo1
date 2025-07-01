import argparse
import os
import xml.etree.ElementTree as ET
from paths import (
    DATA_DIR,
)

#转换.xml到.txt
def xml_to_yolo(xml_path, output_dir, class_map):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    # 获取图像尺寸
    size = root.find('size')
    img_w = int(size.find('width').text)
    img_h = int(size.find('height').text)
    # 生成YOLO标签文件路径
    txt_path = os.path.join(output_dir, os.path.basename(xml_path).replace('.xml', '.txt'))

    with open(txt_path, 'w') as f:
        for obj in root.findall('object'):
            cls_name = obj.find('name').text
            if cls_name not in class_map:
                continue

            bbox = obj.find('bndbox')
            xmin = float(bbox.find('xmin').text)
            ymin = float(bbox.find('ymin').text)
            xmax = float(bbox.find('xmax').text)
            ymax = float(bbox.find('ymax').text)

            # 计算YOLO格式坐标
            x_center = (xmin + xmax) / 2 / img_w
            y_center = (ymin + ymax) / 2 / img_h
            width = (xmax - xmin) / img_w
            height = (ymax - ymin) / img_h

            # 写入文件
            cls_id = class_map.index(cls_name)
            f.write(f"{cls_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

#自动识别类别
def extract_classes(xml_dir):
    class_set = set()
    for xml_file in os.listdir(xml_dir):
        if not xml_file.endswith(".xml"):
            continue
        tree = ET.parse(os.path.join(xml_dir, xml_file))
        root = tree.getroot()
        for obj in root.findall("object"):
            class_name = obj.find("name").text.strip()
            class_set.add(class_name)
    return sorted(list(class_set))


if __name__ == '__main__':

    #定义输入输出文件夹
    xml_dir = DATA_DIR/"raw"/"original_annotations"
    output_dir = DATA_DIR/"raw"/"yolo_staged_labels"
    class_map = []
    #用户手动指定类别
    print("手动指定参考格式如下，如不需要手动指定输入None")
    print('手动输入时，输入END表示结束')
    str = input()
    if str == 'None':
        class_map = extract_classes(xml_dir)
    else:
        for i in range(999):
            if str == 'END':
                break
            class_map.append(str)
            str = input()

    print(class_map)
    for xml_file in os.listdir(xml_dir):
        if xml_file.endswith(".xml"):
            xml_path = os.path.join(xml_dir, xml_file)
            xml_to_yolo(xml_path, output_dir, class_map)