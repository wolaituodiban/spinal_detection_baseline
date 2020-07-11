import json
import math
import os
from multiprocessing import Pool, cpu_count
from typing import Dict, Tuple

import numpy as np
import torch
import torchvision.transforms.functional as tf
from PIL import Image
from tqdm import tqdm

from .dicom_utils import read_one_dcm

PADDING_VALUE: int = 0


def read_dcms(dcm_dir, error_msg=False) -> (Dict[Tuple[str, str, str], Image.Image], Dict[Tuple[str, str, str], dict]):
    """
    读取文件夹内的所有dcm文件
    :param dcm_dir:
    :param error_msg: 是否打印错误信息
    :return: 包含图像信息的字典，和包含元数据的字典
    """
    dcm_paths = []
    for study in os.listdir(dcm_dir):
        study_path = os.path.join(dcm_dir, study)
        for dcm_name in os.listdir(study_path):
            dcm_path = os.path.join(study_path, dcm_name)
            dcm_paths.append(dcm_path)

    with Pool(cpu_count()) as pool:
        async_results = []
        for dcm_path in dcm_paths:
            async_results.append(pool.apply_async(read_one_dcm, (dcm_path,)))

        images, metainfos = {}, {}
        for async_result in tqdm(async_results, ascii=True):
            async_result.wait()
            try:
                metainfo, image = async_result.get()
            except RuntimeError as e:
                if error_msg:
                    print(e)
                continue
            key = metainfo['studyUid'], metainfo['seriesUid'], metainfo['instanceUid']
            del metainfo['studyUid'], metainfo['seriesUid'], metainfo['instanceUid']
            images[key] = tf.to_pil_image(image)
            metainfos[key] = metainfo

    return images, metainfos


def get_spacing(metainfos: Dict[Tuple[str, str, str], dict]) -> Dict[Tuple[str, str, str], torch.Tensor]:
    """
    从元数据中获取像素点间距的信息
    :param metainfos:
    :return:
    """
    output = {}
    for k, v in metainfos.items():
        spacing = v['pixelSpacing']
        spacing = spacing.split('\\')
        spacing = list(map(float, spacing))
        output[k] = torch.tensor(spacing)
    return output


with open(os.path.join(os.path.dirname(__file__), 'static_files/spinal_vertebra_id.json'), 'r') as file:
    SPINAL_VERTEBRA_ID = json.load(file)

with open(os.path.join(os.path.dirname(__file__), 'static_files/spinal_disc_id.json'), 'r') as file:
    SPINAL_DISC_ID = json.load(file)

assert set(SPINAL_VERTEBRA_ID.keys()).isdisjoint(set(SPINAL_DISC_ID.keys()))

with open(os.path.join(os.path.dirname(__file__), 'static_files/spinal_vertebra_disease.json'), 'r') as file:
    SPINAL_VERTEBRA_DISEASE_ID = json.load(file)

with open(os.path.join(os.path.dirname(__file__), 'static_files/spinal_disc_disease.json'), 'r') as file:
    SPINAL_DISC_DISEASE_ID = json.load(file)


def read_annotation(path) -> Dict[Tuple[str, str, str], Tuple[torch.Tensor, torch.Tensor]]:
    """

    :param path:
    :return: 字典的key是（studyUid，seriesUid，instance_uid）
             字典的value是两个矩阵，第一个矩阵对应锥体，第一个矩阵对应椎间盘
             矩阵每一行对应一个脊柱的位置，前两列是位置的坐标(横坐标, 纵坐标)，之后每一列对应一种疾病
             坐标为0代表缺失
             ！注意图片的坐标和tensor的坐标是转置关系的
    """
    with open(path, 'r') as annotation_file:
        # non_hit_count用来统计为被编码的标记的数量，用于预警
        non_hit_count = {}
        annotation = {}
        for x in json.load(annotation_file):
            study_uid = x['studyUid']

            assert len(x['data']) == 1, (study_uid, len(x['data']))
            data = x['data'][0]
            instance_uid = data['instanceUid']
            series_uid = data['seriesUid']

            assert len(data['annotation']) == 1, (study_uid, len(data['annotation']))
            points = data['annotation'][0]['data']['point']

            vertebra_label = torch.full([len(SPINAL_VERTEBRA_ID), 3],
                                        PADDING_VALUE, dtype=torch.long)
            disc_label = torch.full([len(SPINAL_DISC_ID), 3],
                                    PADDING_VALUE, dtype=torch.long)
            for point in points:
                identification = point['tag']['identification']
                if identification in SPINAL_VERTEBRA_ID:
                    position = SPINAL_VERTEBRA_ID[identification]
                    diseases = point['tag']['vertebra']

                    vertebra_label[position, :2] = torch.tensor(point['coord'])
                    for disease in diseases.split(','):
                        if disease in SPINAL_VERTEBRA_DISEASE_ID:
                            disease = SPINAL_VERTEBRA_DISEASE_ID[disease]
                            vertebra_label[position, 2] = disease
                elif identification in SPINAL_DISC_ID:
                    position = SPINAL_DISC_ID[identification]
                    diseases = point['tag']['disc']

                    disc_label[position, :2] = torch.tensor(point['coord'])
                    for disease in diseases.split(','):
                        if disease in SPINAL_DISC_DISEASE_ID:
                            disease = SPINAL_DISC_DISEASE_ID[disease]
                            disc_label[position, 2] = disease
                elif identification in non_hit_count:
                    non_hit_count[identification] += 1
                else:
                    non_hit_count[identification] = 1

            annotation[study_uid, series_uid, instance_uid] = vertebra_label, disc_label
    if len(non_hit_count) > 0:
        print(non_hit_count)
    return annotation


def resize(size: Tuple[int, int], image: Image.Image, spacing: torch.Tensor, *coords: torch.Tensor):
    """

    :param size: [height, width]，height对应纵坐标，width对应横坐标
    :param image: 图像
    :param spacing: 像素点间距
    :param coords: 标注是图像上的坐标，[[横坐标,纵坐标]]，横坐标从左到有，纵坐标从上到下
    :return: resize之后的image，spacing，annotation
    """
    # image.size是[width, height]
    height_ratio = size[0] / image.size[1]
    width_ratio = size[1] / image.size[0]

    ratio = torch.tensor([width_ratio, height_ratio])
    spacing = spacing / ratio
    coords = [coord * ratio for coord in coords]
    image = tf.resize(image, size)

    output = [image, spacing] + coords
    return output


def rotate_point(points: torch.Tensor, angel, center: torch.Tensor) -> torch.Tensor:
    """
    将points绕着center顺时针旋转angel度
    :param points: size of（*， 2）
    :param angel:
    :param center: size of（2，）
    :return:
    """
    if angel == 0:
        return points
    angel = angel * math.pi / 180
    while len(center.shape) < len(points.shape):
        center = center.unsqueeze(0)
    cos = math.cos(angel)
    sin = math.sin(angel)
    rotate_mat = torch.tensor([[cos, -sin], [sin, cos]], dtype=torch.float32, device=points.device)
    output = points - center
    output = torch.matmul(output, rotate_mat)
    return output + center


def rotate_batch(points: torch.Tensor, angels: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
    """
    将一个batch的点，按照不同的角度和中心转旋
    :param points: (num_batch, num_points, 2)
    :param angels: (num_batch,)
    :param centers: (num_batch, 2)
    :return:
    """
    centers = centers.unsqueeze(1)
    output = points - centers

    angels = angels * math.pi / 180
    cos = angels.cos()
    sin = angels.sin()
    rotate_mats = torch.stack([cos, sin, -sin, cos], dim=1).reshape(angels.shape[0], 1, 2, 2)
    output = output.unsqueeze(-1)
    output = output * rotate_mats
    output = output.sum(dim=-1)
    return output + centers


def rotate(image: Image.Image, points: torch.Tensor, angel: int) -> (Image.Image, torch.Tensor):
    center = torch.tensor(image.size, dtype=torch.float32) / 2
    return tf.rotate(image, angel), rotate_point(points, angel, center)


def gen_distmap(image: torch.Tensor, spacing: torch.Tensor, *gt_coords: torch.Tensor, angel=0):
    """
    先将每个像素点的坐标顺时针旋转angel之后，再计算到标注像素点的物理距离
    :param image: height * weight
    :param gt_coords: size of（*， 2）
    :param spacing:
    :param angel: 
    :return:
    """
    coord = torch.where(image.squeeze() < np.inf)
    # 注意需要反转横纵坐标
    center = torch.tensor([image.shape[2], image.shape[1]], dtype=torch.float32) / 2
    coord = torch.stack(coord[::-1], dim=1).reshape(image.size(1), image.size(2), 2)
    coord = rotate_point(coord, angel, center)
    dists = []
    for gt_coord in gt_coords:
        gt_coord = rotate_point(gt_coord, angel, center)
        dist = []
        for point in gt_coord:
            dist.append((((coord - point) * spacing) ** 2).sum(dim=-1).sqrt())
        dist = torch.stack(dist, dim=0)
        dists.append(dist)
    if len(dists) == 1:
        return dists[0]
    else:
        return dists


def gen_mask(coord: torch.Tensor):
    return (coord.index_select(-1, torch.arange(2, device=coord.device)) != PADDING_VALUE).any(dim=-1)
