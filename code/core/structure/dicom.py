import random

import SimpleITK as sitk
import numpy as np
import torch
import torchvision.transforms.functional as tf
from PIL import Image

from ..data_utils import resize, rotate, gen_distmap
from ..dicom_utils import DICOM_TAG


def lazy_property(func):
    attr_name = "_lazy_" + func.__name__

    @property
    def _lazy_property(self):
        if not hasattr(self, attr_name):
            setattr(self, attr_name, func(self))

        return getattr(self, attr_name)

    return _lazy_property


def str2tensor(s: str) -> torch.Tensor:
    """

    :param s: numbers separated by '\\', eg.  '0.71875\\0.71875 '
    :return: 1-D tensor
    """
    return torch.tensor(list(map(float, s.split('\\'))))


def unit_vector(tensor: torch.Tensor, dim=-1):
    norm = (tensor ** 2).sum(dim=dim, keepdim=True).sqrt()
    return tensor / norm


def unit_normal_vector(orientation: torch.Tensor):
    temp1 = orientation[:, [1, 2, 0]]
    temp2 = orientation[:, [2, 0, 1]]
    output = temp1 * temp2[[1, 0]]
    output = output[0] - output[1]
    return unit_vector(output, dim=-1)


class DICOM:
    """
    解析dicom文件
    属性：
        study_uid：检查ID
        series_uid：序列ID
        instance_uid：图像ID
        series_description：序列描述，用于区分T1、T2等
        pixel_spacing: 长度为2的向量，像素的物理距离，单位是毫米
        image_position：长度为3的向量，图像左上角在人坐标系上的坐标，单位是毫米
        image_orientation：2x3的矩阵，第一行表示图像从左到右的方向，第二行表示图像从上到下的方向，单位是毫米？
        unit_normal_vector: 长度为3的向量，图像的单位法向量，单位是毫米？
        image：PIL.Image.Image，图像
    注：人坐标系，规定人体的左边是X轴的方向，从面部指向背部的方向表示y轴的方向，从脚指向头的方向表示z轴的方向
    """

    def __init__(self, file_path):
        self.file_path = file_path
        self.error_msg = ''

        reader = sitk.ImageFileReader()
        reader.LoadPrivateTagsOn()
        reader.SetImageIO('GDCMImageIO')
        reader.SetFileName(file_path)
        try:
            reader.ReadImageInformation()
        except RuntimeError:
            pass

        try:
            self.study_uid = reader.GetMetaData(DICOM_TAG['studyUid'])
        except RuntimeError:
            self.study_uid = ''

        try:
            self.series_uid: str = reader.GetMetaData(DICOM_TAG['seriesUid'])
        except RuntimeError:
            self.series_uid = ''

        try:
            self.instance_uid: str = reader.GetMetaData(DICOM_TAG['instanceUid'])
        except RuntimeError:
            self.instance_uid = ''

        try:
            self.series_description: str = reader.GetMetaData(DICOM_TAG['seriesDescription'])
        except RuntimeError:
            self.series_description = ''

        try:
            self._pixel_spacing = reader.GetMetaData(DICOM_TAG['pixelSpacing'])
        except RuntimeError:
            self._pixel_spacing = None

        try:
            self._image_position = reader.GetMetaData(DICOM_TAG['imagePosition'])
        except RuntimeError:
            self._image_position = None

        try:
            self._image_orientation = reader.GetMetaData(DICOM_TAG['imageOrientation'])
        except RuntimeError:
            self._image_orientation = None

        try:
            image = reader.Execute()
            if image.GetNumberOfComponentsPerPixel() == 1:
                image = sitk.RescaleIntensity(image, 0, 255)
                if reader.GetMetaData('0028|0004').strip() == 'MONOCHROME1':
                    image = sitk.InvertIntensity(image, maximum=255)
                image = sitk.Cast(image, sitk.sitkUInt8)
            img_x = sitk.GetArrayFromImage(image)[0]
            self.image: Image.Image = tf.to_pil_image(img_x)
        except RuntimeError:
            self.image = None

    @lazy_property
    def pixel_spacing(self):
        if self._pixel_spacing is None:
            return torch.full([2, ], fill_value=np.nan)
        else:
            return str2tensor(self._pixel_spacing)

    @lazy_property
    def image_position(self):
        if self._image_position is None:
            return torch.full([3, ], fill_value=np.nan)
        else:
            return str2tensor(self._image_position)

    @lazy_property
    def image_orientation(self):
        if self._image_orientation is None:
            return torch.full([2, 3], fill_value=np.nan)
        else:
            return unit_vector(str2tensor(self._image_orientation).reshape(2, 3))

    @lazy_property
    def unit_normal_vector(self):
        if self.image_orientation is None:
            return torch.full([3, ], fill_value=np.nan)
        else:
            return unit_normal_vector(self.image_orientation)

    @lazy_property
    def t_type(self):
        if 'T1' in self.series_description.upper():
            return 'T1'
        elif 'T2' in self.series_description.upper():
            return 'T2'
        else:
            return None

    @lazy_property
    def plane(self):
        if torch.isnan(self.unit_normal_vector).all():
            return None
        elif torch.matmul(self.unit_normal_vector, torch.tensor([0., 0., 1.])).abs() > 0.75:
            # 轴状位，水平切开
            return 'transverse'
        elif torch.matmul(self.unit_normal_vector, torch.tensor([1., 0., 0.])).abs() > 0.75:
            # 矢状位，左右切开
            return 'sagittal'
        elif torch.matmul(self.unit_normal_vector, torch.tensor([0., 1., 0.])).abs() > 0.75:
            # 冠状位，前后切开
            return 'coronal'
        else:
            # 不知道
            return None

    @lazy_property
    def mean(self):
        if self.image is None:
            return None
        else:
            return tf.to_tensor(self.image).mean()

    @property
    def size(self):
        """

        :return: width and height
        """
        if self.image is None:
            return None
        else:
            return self.image.size

    def pixel_coord2human_coord(self, coord: torch.Tensor) -> torch.Tensor:
        """
        将图像上的像素坐标转换成人坐标系上的坐标
        :param coord: 像素坐标，Nx2的矩阵或者长度为2的向量
        :return: 人坐标系坐标，Nx3的矩阵或者长度为3的向量
        """
        return torch.matmul(coord * self.pixel_spacing, self.image_orientation) + self.image_position

    def point_distance(self, human_coord: torch.Tensor) -> torch.Tensor:
        """
        点到图像平面的距离，单位为毫米
        :param human_coord: 人坐标系坐标，Nx3的矩阵或者长度为3的向量
        :return: 长度为N的向量或者标量
        """
        return torch.matmul(human_coord - self.image_position, self.unit_normal_vector).abs()

    def projection(self, human_coord: torch.Tensor) -> torch.Tensor:
        """
        将人坐标系中的点投影到图像上，并输出像素坐标
        :param human_coord: 人坐标系坐标，Nx3的矩阵或者长度为3的向量
        :return:像素坐标，Nx2的矩阵或者长度为2的向量
        """
        cos = torch.matmul(human_coord - self.image_position, self.image_orientation.transpose(0, 1))
        return (cos / self.pixel_spacing).round()

    def transform(self, pixel_coord: torch.Tensor,
                  size=None, prob_rotate=0, max_angel=0, distmap=False, tensor=True) -> (torch.Tensor, torch.Tensor):
        """
        返回image tensor和distance map
        :param pixel_coord:
        :param size:
        :param prob_rotate:
        :param max_angel:
        :param distmap: 是否返回distmap
        :param tensor: 如果True，那么返回图片的tensor，否则返回Image
        :return:
        """
        image, pixel_spacing = self.image, self.pixel_spacing
        if size is not None:
            image, pixel_spacing, pixel_coord = resize(size, image, pixel_spacing, pixel_coord)

        if max_angel > 0 and random.random() <= prob_rotate:
            angel = random.randint(-max_angel, max_angel)
            image, pixel_coord = rotate(image, pixel_coord, angel)

        if tensor:
            image = tf.to_tensor(image)
        pixel_coord = pixel_coord.round().long()
        if distmap:
            distmap = gen_distmap(image, pixel_spacing, pixel_coord)
            return image, pixel_coord, distmap
        else:
            return image, pixel_coord
