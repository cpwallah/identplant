import json
import os
import random
import string
import time

import torch
from PIL import Image
# from gevent import pywsgi
from flask import Flask, request, send_file
from torchvision import transforms

from backbone import resnet50_fpn_backbone
from config import SQLManager
from draw_box_utils import draw_objs
from network_files import FasterRCNN

# from torchvision import transforms

app = Flask(__name__)


def generate_code(length):
    code = ''
    for i in range(length):
        code += random.choice(string.ascii_uppercase + string.digits)
    return code


def create_model(num_classes):
    # 注意，这里的norm_layer要和训练脚本中保持一致
    backbone = resnet50_fpn_backbone(norm_layer=torch.nn.BatchNorm2d)
    model = FasterRCNN(backbone=backbone, num_classes=num_classes, rpn_score_thresh=0.5)
    return model


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


db = SQLManager()
citys = db.get_list('select * from diseasebrief ')
print(citys)
db.close()



# get devices
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("using {} device.".format(device))

# create model
model = create_model(num_classes=21)

# load train weights
# weights_path = "./model/best.pt"
weights_path = "./model/PENETBFNet-model-11.pth"
assert os.path.exists(weights_path), "{} file dose not exist.".format(weights_path)
weights_dict = torch.load(weights_path, map_location='cpu')
weights_dict = weights_dict["model"]
# for (key, value) in weights_dict.items():
#     print('key: ', key, 'value: ', value)
weights_dict.pop('roi_heads.box_predictor.cls_score.weight')
weights_dict.pop('roi_heads.box_predictor.cls_score.bias')
weights_dict.pop('roi_heads.box_predictor.bbox_pred.weight')
weights_dict.pop('roi_heads.box_predictor.bbox_pred.bias')
model.load_state_dict(weights_dict, strict=False)
model.to(device)

# read class_indict
label_json_path = './pascal_voc_classes_cn.json'
assert os.path.exists(label_json_path), "json file {} dose not exist.".format(label_json_path)
with open(label_json_path, 'r', encoding='utf-8') as f:
    class_dict = json.load(f)

category_index = {str(v): str(k) for k, v in class_dict.items()}


@app.route("/static/<string:filename>")
def get_filename(filename):
    filename = basedir.replace('\\', '/') + '/static/' + filename
    return send_file(filename, mimetype='image/jpeg')


basedir = os.path.abspath(os.path.dirname(__file__))


@app.route('/classify', methods=['POST'])
def classify():
    # 从请求中获取图像文件
    file = request.files['image']
    result_name = str(int(round(time.time() * 1000))) + generate_code(5)

    save_path = basedir.replace('\\', '/') + '/static/images/' + result_name + "-original.jpg"
    print("original image path : " + save_path)

    original_img = Image.open(file.stream)

    original_img.save(save_path)

    data_transform = transforms.Compose([transforms.ToTensor()])
    img = data_transform(original_img)
    img = torch.unsqueeze(img, dim=0)

    model.eval()  # 进入验证模式
    with torch.no_grad():
        # init
        img_height, img_width = img.shape[-2:]
        init_img = torch.zeros((1, 3, img_height, img_width), device=device)
        model(init_img)
        predictions = model(img.to(device))[0]
        predict_boxes = predictions["boxes"].to("cpu").numpy()
        predict_classes = predictions["labels"].to("cpu").numpy()
        predict_scores = predictions["scores"].to("cpu").numpy()
        plot_img = draw_objs(original_img,
                             predict_boxes,
                             predict_classes,
                             predict_scores,
                             category_index=category_index,
                             box_thresh=0.1,
                             line_thickness=1,
                             font='arial.ttf',
                             font_size=15)
        save_path = basedir.replace('\\', '/') + '/static/images/' + result_name + ".jpg"
        print("result image path :" + save_path)
        plot_img.save(save_path)
    return {
        'code': 200,
        'path': request.host_url + 'static/images/' + result_name + ".jpg"
    }


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
    # server = pywsgi.WSGIServer(('127.0.0.1', 5000), app)
    # server.serve_forever()
