import json
import os
import random
import string
import time

import torch
from PIL import Image
# from gevent import pywsgi
from flask import Flask, request, send_file, send_from_directory
from torchvision import transforms

from backbone import resnet50_fpn_backbone
from config import SQLManager
from draw_box_utils import draw_objs
from network_files import FasterRCNN

# from torchvision import transforms

app = Flask(__name__)
basedir = os.path.abspath(os.path.dirname(__file__))


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


@app.route("/static/<string:filename>")
def get_filename(filename):
    filename = basedir.replace('\\', '/') + '/static/' + filename
    return send_file(filename, mimetype='image/jpeg')


@app.route("/plant/types")
def get_plants():
    db = SQLManager()
    result = db.get_list('SELECT `name`,`fname`,`disease_id` FROM diseasebrief')
    db.close()
    return {'code': 200, 'rows': result}


@app.route("/plantById/<string:disease_id>")
def get_disease_by_id(disease_id):
    db = SQLManager()
    result = db.get_one('SELECT * FROM diseasebrief WHERE disease_id = ' + disease_id)
    db.close()
    return {'code': 200, 'rows': result}


@app.route("/plantByName/<string:name>")
def get_disease_by_name(name):
    db = SQLManager()
    result = db.get_one('SELECT * FROM diseasebrief WHERE `name` = ' + "'" + name + "'")
    db.close()
    return {'code': 200, 'rows': result}


@app.route("/simpleImg")
def get_simple_img():
    path = basedir.replace('\\', '/') + '/static/simpleImg'  # 文件夹目录
    files = os.listdir(path)  # 得到文件夹下的所有文件名称
    return {'code': 200, 'rows': files}


@app.route('/download/android')
def download():
    print('downloading ...')
    return send_from_directory(basedir.replace('\\', '/') + '/static/androidapk', "release.apk", as_attachment=True)


@app.route("/flask/login", methods=['POST'])
def login():
    data = request.get_data()
    data = json.loads(data)
    email = data['email']
    password = data['password']
    db = SQLManager()
    user = db.get_one('SELECT * FROM `user` WHERE email = ' + "'" + email + "'")
    if user['password'] == password:
        return {'code': 200, 'user': user}
    else:
        return {'code': 401, 'msg': '账号或密码错误'}


@app.route("/flask/register", methods=['POST'])
def register():
    data = request.get_data()
    data = json.loads(data)
    email = data['email']
    password = data['password']
    db = SQLManager()
    user = db.get_one('SELECT * FROM `user` WHERE email = ' + "'" + email + "'")
    if user is None:
        db.create("INSERT INTO `user` (`username`, `password`, `email`)  VALUE('%s','%s','%s')" \
                  % (email, password, email))
        user = {'email': email, 'password': password, 'username': email}
        return {'code': 200, 'user': user}
    else:
        return {'code': 401, 'msg': '该邮箱已注册'}


@app.route('/classify', methods=['POST'])
def classify():
    # 从请求体中获取图像文件
    file = request.files['image']
    uid = request.form.get("uid")
    print(uid)
    # 生成随机文件名
    result_name = str(int(round(time.time() * 1000))) + generate_code(5)
    # 保存路径
    save_path = basedir.replace('\\', '/') + '/static/images/' + result_name + "-original.jpg"
    print("original image path : " + save_path)

    original_img = Image.open(file.stream)

    original_img.save(save_path)

    data_transform = transforms.Compose([transforms.ToTensor()])
    img = data_transform(original_img)
    img = torch.unsqueeze(img, dim=0)

    # get devices
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # create model
    model = create_model(num_classes=39)

    # load train weights
    # weights_path = "./model/best.pt"
    weights_path = "./model/PENETBFNet-model-11.pth"
    assert os.path.exists(weights_path), "{} file dose not exist.".format(weights_path)
    weights_dict = torch.load(weights_path, map_location='cpu')
    weights_dict = weights_dict["model"]
    # for (key, value) in weights_dict.items():
    #     print('key: ', key, 'value: ', value)
    # weights_dict.pop('roi_heads.box_predictor.cls_score.weight')
    # weights_dict.pop('roi_heads.box_predictor.cls_score.bias')
    # weights_dict.pop('roi_heads.box_predictor.bbox_pred.weight')
    # weights_dict.pop('roi_heads.box_predictor.bbox_pred.bias')
    model.load_state_dict(weights_dict, strict=False)
    model.to(device)

    # read class_indict
    label_json_path = './pascal_voc_classes_cn.json'
    assert os.path.exists(label_json_path), "json file {} dose not exist.".format(label_json_path)
    with open(label_json_path, 'r', encoding='utf-8') as f:
        class_dict = json.load(f)
    print('label after')
    category_index = {str(v): str(k) for k, v in class_dict.items()}

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
        draw_result = draw_objs(original_img,
                                predict_boxes,
                                predict_classes,
                                predict_scores,
                                category_index=category_index,
                                box_thresh=0.1,
                                line_thickness=1,
                                font='arial.ttf',
                                font_size=15)
        # plot_img = draw_result[0]
        # save_path = basedir.replace('\\', '/') + '/static/images/' + result_name + ".jpg"
        # print("result image path :" + save_path)
        # plot_img.save(save_path)

    final = {
        'code': 200,
        'name': draw_result[0],
        'color': draw_result[1],
        'acc': 1
        # 'path': request.host_url + 'static/images/' + result_name + ".jpg"
    }
    print(final)
    return final


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
    # server = pywsgi.WSGIServer(('127.0.0.1', 5000), app)
    # server.serve_forever()
