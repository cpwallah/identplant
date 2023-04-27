import os

from PIL.Image import Image, fromarray
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont
from PIL import ImageColor
import numpy as np
import random
from torch._appdirs import unicode

STANDARD_COLORS = [
    'AliceBlue', 'Chartreuse', 'Aqua', 'Aquamarine', 'Azure', 'Beige', 'Bisque',
    'BlanchedAlmond', 'BlueViolet', 'BurlyWood', 'CadetBlue', 'AntiqueWhite',
    'Chocolate', 'Coral', 'CornflowerBlue', 'Cornsilk', 'Crimson', 'Cyan',
    'DarkCyan', 'DarkGoldenRod', 'DarkGrey', 'DarkKhaki', 'DarkOrange',
    'DarkOrchid', 'DarkSalmon', 'DarkSeaGreen', 'DarkTurquoise', 'DarkViolet',
    'DeepPink', 'DeepSkyBlue', 'DodgerBlue', 'FireBrick', 'FloralWhite',
    'ForestGreen', 'Fuchsia', 'Gainsboro', 'GhostWhite', 'Gold', 'GoldenRod',
    'Salmon', 'Tan', 'HoneyDew', 'HotPink', 'IndianRed', 'Ivory', 'Khaki',
    'Lavender', 'LavenderBlush', 'LawnGreen', 'LemonChiffon', 'LightBlue',
    'LightCoral', 'LightCyan', 'LightGoldenRodYellow', 'LightGray', 'LightGrey',
    'LightGreen', 'LightPink', 'LightSalmon', 'LightSeaGreen', 'LightSkyBlue',
    'LightSlateGray', 'LightSlateGrey', 'LightSteelBlue', 'LightYellow', 'Lime',
    'LimeGreen', 'Linen', 'Magenta', 'MediumAquaMarine', 'MediumOrchid',
    'MediumPurple', 'MediumSeaGreen', 'MediumSlateBlue', 'MediumSpringGreen',
    'MediumTurquoise', 'MediumVioletRed', 'MintCream', 'MistyRose', 'Moccasin',
    'NavajoWhite', 'OldLace', 'Olive', 'OliveDrab', 'Orange', 'OrangeRed',
    'Orchid', 'PaleGoldenRod', 'PaleGreen', 'PaleTurquoise', 'PaleVioletRed',
    'PapayaWhip', 'PeachPuff', 'Peru', 'Pink', 'Plum', 'PowderBlue', 'Purple',
    'Red', 'RosyBrown', 'RoyalBlue', 'SaddleBrown', 'Green', 'SandyBrown',
    'SeaGreen', 'SeaShell', 'Sienna', 'Silver', 'SkyBlue', 'SlateBlue',
    'SlateGray', 'SlateGrey', 'Snow', 'SpringGreen', 'SteelBlue', 'GreenYellow',
    'Teal', 'Thistle', 'Tomato', 'Turquoise', 'Violet', 'Wheat', 'White',
    'WhiteSmoke', 'Yellow', 'YellowGreen'
]


# RGB格式颜色转换为16进制颜色格式
def RGB_to_Hex(rgb):
    RGB = [rgb[0], rgb[1], rgb[2]]  # 将RGB格式划分开来
    color = '#'
    for i in RGB:
        num = int(i)
        # 将R、G、B分别转化为16进制拼接转换并大写  hex() 函数用于将10进制整数转换成16进制，以字符串形式表示
        color += str(hex(num))[-2:].replace('x', '0').upper()
    print(color)
    return color


def draw_objs(image: Image,
              boxes: np.ndarray = None,
              classes: np.ndarray = None,
              scores: np.ndarray = None,
              masks: np.ndarray = None,
              category_index: dict = None,
              box_thresh: float = 0.1,
              mask_thresh: float = 0.5,
              line_thickness: int = 8,
              font: str = 'arial.ttf',
              font_size: int = 24,
              draw_boxes_on_image: bool = True,
              draw_masks_on_image: bool = False):
    """
    将目标边界框信息，类别信息，mask信息绘制在图片上
    Args:
        image: 需要绘制的图片
        boxes: 目标边界框信息
        classes: 目标类别信息
        scores: 目标概率信息
        masks: 目标mask信息
        category_index: 类别与名称字典
        box_thresh: 过滤的概率阈值
        mask_thresh:
        line_thickness: 边界框宽度
        font: 字体类型
        font_size: 字体大小
        draw_boxes_on_image:
        draw_masks_on_image:

    Returns:
    """

    # 过滤掉低概率的目标
    idxs = np.greater(scores, box_thresh)
    boxes = boxes[idxs]
    classes = classes[idxs]
    scores = scores[idxs]
    print(classes)
    print(scores)

    if masks is not None:
        masks = masks[idxs]
    if len(boxes) == 0:
        # print("低目标")
        return image

    co = random.randint(0, len(STANDARD_COLORS)-1)
    color = ImageColor.getrgb(STANDARD_COLORS[co])

    result_str = ''
    result_color = None
    if draw_boxes_on_image:
        for box, cls, score in zip(boxes, classes, scores):
            # draw_text(draw, box.tolist(), int(cls), float(score), category_index, color, font, font_size)
            result_str = f"{category_index[str(cls)]}"
            result_color = color
            break
        # tempIndex = tempIndex + 1

    return [result_str, RGB_to_Hex(result_color)]
