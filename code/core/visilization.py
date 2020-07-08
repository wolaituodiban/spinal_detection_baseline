from typing import Union
import torch
import torchvision.transforms.functional as tf
from PIL import Image


def to_tensor(image):
    if isinstance(image, Image.Image):
        return tf.to_tensor(image)
    else:
        return image.clone().detach()


def visilize_coord(image: Union[Image.Image, torch.Tensor], *coords: torch.Tensor, _range=10) -> Image.Image:
    """
    关于annotation的结构请参考read_annotation
    :param image:
    :param coords:
    :param _range:
    :return:
    """
    image = to_tensor(image)
    for coord in coords:
        for point in coord:
            # 注意，image和tensor存在转置关系
            image[0, int(point[1]-_range):int(point[1]+_range), int(point[0]-_range):int(point[0]+_range)] = 0
    return tf.to_pil_image(image)


def visilize_distmap(image: Union[Image.Image, torch.Tensor], *distmaps: torch.Tensor, max_dist=8) -> Image.Image:
    """
    关于label的结构请参考gen_label
    :param image:
    :param distmaps:
    :param max_dist:
    :return:
    """
    image = to_tensor(image)
    for distmap in distmaps:
        image[(distmap < max_dist).sum(dim=0).bool().unsqueeze(0)] = 0
    return tf.to_pil_image(image)


def visilize_annotation(image, *annotations, _range=10):
    image = to_tensor(image)
    for annotation in annotations:
        for point in annotation['data'][0]['annotation'][0]['data']['point']:
            coord = point['coord']
            image[0, int(coord[1]-_range):int(coord[1]+_range), int(coord[0]-_range):int(coord[0]+_range)] = 0
    return tf.to_pil_image(image)
