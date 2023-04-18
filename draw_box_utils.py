import os

from PIL.Image import Image, fromarray
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont
from PIL import ImageColor
import numpy as np
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


def draw_text(draw,
              box: list,
              cls: int,
              score: float,
              category_index: dict,
              color: str,
              font: str = 'arial.ttf',
              font_size: int = 24):
    """
    将目标边界框和类别信息绘制到图片上
    """
    try:
        font = ImageFont.truetype(font, font_size)
    except IOError:
        font = ImageFont.load_default()

    left, top, right, bottom = box
    # If the total height of the display strings added to the top of the bounding
    # box exceeds the top of the image, stack the strings below the bounding box
    # instead of above.

    # {category_index[str(cls)]}
    # display_str = f"{category_index[str(cls)]}: {int(100 * score)}%"
    display_str = f"{category_index[str(cls)]}"
    # display_str_heights = [font.getsize(ds)[1] for ds in display_str]
    # Each display_str has a top and bottom margin of 0.05x.
    # display_str_height = (1 + 2 * 0.05) * max(display_str_heights)

    # if top > display_str_height:
    #     text_top = top - display_str_height
    #     text_bottom = top
    # else:
    #     text_top = bottom
    #     text_bottom = bottom + display_str_height

    # font_path = os.path.join("assets", "Hiragino Sans GB.ttc")

    # local_font = ImageFont.truetype("simsun", size=font_size, encoding="gbk")
    # for ds in display_str:
    #     # text_width, text_height = font.getsize(ds)
    #     # text_width = text_width * 1.5
    #     # margin = np.ceil(0.05 * text_width)
    #     # draw.rectangle([(left, text_top),
    #     #                 (left + text_width + 2 * margin, text_bottom)], fill=color)
    #     # draw.text((left + margin, text_top),
    #     #           ds,
    #     #           fill='black',
    #     #           font=local_font)
    #     left += text_width
    return display_str


def draw_masks(image, masks, colors, thresh: float = 0.7, alpha: float = 0.5):
    np_image = np.array(image)
    masks = np.where(masks > thresh, True, False)

    # colors = np.array(colors)
    img_to_draw = np.copy(np_image)
    # TODO: There might be a way to vectorize this
    for mask, color in zip(masks, colors):
        img_to_draw[mask] = color

    out = np_image * (1 - alpha) + img_to_draw * alpha
    return fromarray(out.astype(np.uint8))



# RGB格式颜色转换为16进制颜色格式
def RGB_to_Hex(rgb):
    RGB = [rgb[0],rgb[1],rgb[2]]           # 将RGB格式划分开来
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
    if masks is not None:
        masks = masks[idxs]
    if len(boxes) == 0:
        # print("低目标")
        return image

    colors = [ImageColor.getrgb(STANDARD_COLORS[cls % len(STANDARD_COLORS)]) for cls in classes]

    # print(boxes)
    # if draw_boxes_on_image:
    #     # draw = ImageDraw.Draw(image)
    #     for box, cls, score, color in zip(boxes, classes, scores, colors):
    #         left, top, right, bottom = box
    #         maxScore = maxScore if maxScore > score else score
    #         index = index + 1
    # 绘制目标边界框
    # draw.line([(left, top), (left, bottom), (right, bottom),
    #            (right, top), (left, top)], width=line_thickness, fill=color)
    # # 绘制类别和概率信息
    # draw_text(draw, box.tolist(), int(cls), float(score), category_index, color, font, font_size)
    # print(index)
    # draw = ImageDraw.Draw(image)
    # box = boxes[index]
    # left, top, right, bottom = box
    # score = scores[index]
    # color = colors[index]
    # cls = classes[index]
    #
    # # 绘制目标边界框
    # draw.line([(left, top), (left, bottom), (right, bottom),
    #            (right, top), (left, top)], width=line_thickness, fill=color)
    # # 绘制类别和概率信息
    # draw_text(draw, box.tolist(), int(cls), float(score), category_index, color, font, font_size)
    # image = draw_masks(image, masks, colors, mask_thresh)
    tempIndex = 0
    result_str = ''
    result_color = None
    if draw_boxes_on_image:
        # Draw all boxes onto image.
        draw = ImageDraw.Draw(image)
        for box, cls, score, color in zip(boxes, classes, scores, colors):
            # if tempIndex == index:
            # left, top, right, bottom = box
            left = 0
            top = 0
            right = image.width - line_thickness
            bottom = image.height - line_thickness
            # print(box.tolist())
            boxList = [left, 2 + font_size, right, bottom]
            print(boxList)
            # 绘制目标边界框
            draw.line([(left, top), (left, bottom), (right, bottom),
                       (right, top), (left, top)], width=line_thickness, fill=color)
            # 绘制类别和概率信息

            # draw_text(draw, box.tolist(), int(cls), float(score), category_index, color, font, font_size)
            result_str = draw_text(draw, boxList, int(cls), float(score), category_index, color, font, font_size)
            result_color = color
            break
        # tempIndex = tempIndex + 1
    if draw_masks_on_image and (masks is not None):
        # Draw all mask onto image.
        image = draw_masks(image, masks, colors, mask_thresh)

    return [result_str, RGB_to_Hex(result_color)]
