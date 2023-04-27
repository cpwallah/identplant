import json
import os

import torch
from PIL import Image
from torchvision import transforms

from backbone import resnet50_fpn_backbone
from draw_box_utils import draw_objs
from network_files import FasterRCNN
import base64
from io import BytesIO

# from gevent import pywsgi

m_model = None
device = None
category_index = None


def create_model(num_classes):
    global device, m_model, category_index
    # 注意，这里的norm_layer要和训练脚本中保持一致
    backbone = resnet50_fpn_backbone(norm_layer=torch.nn.BatchNorm2d)
    global m_model
    m_model = FasterRCNN(backbone=backbone, num_classes=num_classes, rpn_score_thresh=0.5)
    return m_model


def init_model():
    # get devices
    global device, m_model, category_index
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # create m_model
    m_model = create_model(num_classes=39)
    weights_path = "./model/PENETBFNet-model-11.pth"
    assert os.path.exists(weights_path), "{} file dose not exist.".format(weights_path)
    weights_dict = torch.load(weights_path, map_location='cpu')
    weights_dict = weights_dict["model"]
    m_model.load_state_dict(weights_dict, strict=False)
    m_model.to(device)
    # read class_indict
    label_json_path = './pascal_voc_classes_cn.json'
    assert os.path.exists(label_json_path), "json file {} dose not exist.".format(label_json_path)
    with open(label_json_path, 'r', encoding='utf-8') as f:
        class_dict = json.load(f)
    print('label after')
    category_index = {str(v): str(k) for k, v in class_dict.items()}


def eval_img(img, image_type):
    global device, m_model, category_index
    original_img = None
    if m_model is None:
        init_model()
    if image_type == 'path':
        # 文件路径
        original_img = Image.open(img)
    elif image_type == 'base64':
        # 解码base64字符串并创建图像对象
        decoded_image_data = base64.b64decode(img)
        original_img = Image.open(BytesIO(decoded_image_data))
    elif image_type == 'file':
        original_img = Image.open(img.stream)
    data_transform = transforms.Compose([transforms.ToTensor()])
    img = data_transform(original_img)
    img = torch.unsqueeze(img, dim=0)

    m_model.eval()  # 进入验证模式
    with torch.no_grad():
        # init
        img_height, img_width = img.shape[-2:]
        init_img = torch.zeros((1, 3, img_height, img_width), device=device)
        m_model(init_img)
        predictions = m_model(img.to(device))[0]
        predict_boxes = predictions["boxes"].to("cpu").numpy()
        predict_classes = predictions["labels"].to("cpu").numpy()
        predict_scores = predictions["scores"].to("cpu").numpy()
        draw_result = draw_objs(original_img,
                                predict_boxes,
                                predict_classes,
                                predict_scores,
                                category_index=category_index,
                                box_thresh=0.1,
                                line_thickness=1,
                                font='arial.ttf',
                                font_size=15)
    result = str({"color": draw_result[1], "name": draw_result[0]})
    print(result)
    return result
