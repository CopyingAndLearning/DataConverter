import os
import xml.etree.ElementTree as ET
from PIL import Image
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

object_id = 0
pd.set_option('display.max_columns', None)

# 获取编码对象
def getLabelEncoder():
    return LabelEncoder()

# 读取图像
def getImg(path):
    # 打开一个图像文件
    image_path = path
    image = Image.open(image_path)
    return image

# 同时写入
def toParquet(df,parquet_path):
    # # 将DataFrame转换为Table
    table = pa.Table.from_pandas(df)
    # 写入Parquet文件
    pq.write_table(table, parquet_path)

def parse_xml_to_json(xml_file, image_id, image, labelEncoder):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    data = {
        'image_id': image_id,
        'image': image,
        'width': 0,
        'height': 0,
        'objects': {
            'id': [],
            'area': [],
            'bbox': [],
            'category': []
        }
    }

    # 提取图像尺寸
    data['width'] = int(root.find('size/width').text)
    data['height'] = int(root.find('size/height').text)

    cat_list = []
    # 遍历所有对象
    for obj_idx, obj in enumerate(root.findall('object'), start=1):
        global object_id

        # 提取边界框信息
        bndbox = obj.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)

        # 类别提取
        ## 1、先将所有标签融到一个数组当中；2、转换；
        cat_list.append(obj.find('name').text)

        # 计算面积
        area = (xmax - xmin) * (ymax - ymin)

        # 添加到数据结构中
        data['objects']['id'].append(object_id)
        data['objects']['area'].append(area)
        data['objects']['bbox'].append([xmin, ymin, xmax, ymax])

        # 处理完之后+1
        object_id += 1

    integer_encoded = labelEncoder.fit_transform(cat_list)
    data['objects']['category'] = list(integer_encoded)

    return data

# 遍历文件夹下的所有xml文件
def iterXmlandImg(path, Imgpath, parquet_path):
    xml_folder = path
    img_folder = Imgpath
    le = getLabelEncoder()    # 保证是同一个le
    obj_list = []
    for xml_path,img_path in zip(os.listdir(xml_folder),os.listdir(img_folder)):
        if xml_path.endswith('.xml'):
            xml_file_path = os.path.join(xml_folder, xml_path)
            img_file_path = os.path.join(img_folder, img_path)
            object_data = parse_xml_to_json(xml_file_path, image_id=xml_path, image=getImg(img_file_path), labelEncoder=le)
            # 为每个XML文件的数据添加一个标识符
            obj_list.append(object_data)

    df = pd.DataFrame(obj_list)
    for index, row in df.iterrows():
        image_bytes = row['image'].tobytes()
        df.at[index, 'image'] = image_bytes
    print(df)

    toParquet(df,parquet_path)


def main():
    xml_folder =  "./xml"
    img_folder = "./image"
    parquet_path = 'validation.parquet'
    iterXmlandImg(xml_folder, img_folder, parquet_path)

if __name__ == '__main__':
    main()