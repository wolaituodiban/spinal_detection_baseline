import torch
from collections import Counter
from typing import List
from .dicom import DICOM, lazy_property


class Series(list):
    """
    将DICOM的序列，并且会按照dim的方向，根据image_position对DICOM进行排列
    """
    def __init__(self, dicom_list: List[DICOM]):
        planes = [dicom.plane for dicom in dicom_list]
        plane_counter = Counter(planes)
        self.plane = plane_counter.most_common(1)[0][0]

        if self.plane == 'transverse':
            dim = 2
        elif self.plane == 'sagittal':
            dim = 0
        elif self.plane == 'transverse':
            dim = 1
        else:
            dim = None

        dicom_list = [dicom for dicom in dicom_list if dicom.plane == self.plane]
        if dim is not None:
            dicom_list = sorted(dicom_list, key=lambda x: x.image_position[dim], reverse=True)
        super().__init__(dicom_list)
        self.instance_uids = {d.instance_uid: i for i, d in enumerate(self)}
        self.middle_frame_uid = None

    def __getitem__(self, item) -> DICOM:
        if isinstance(item, str):
            item = self.instance_uids[item]
        return super().__getitem__(item)

    @lazy_property
    def t_type(self):
        t_type_counter = Counter([d.t_type for d in self])
        return t_type_counter.most_common(1)[0][0]

    @lazy_property
    def mean(self):
        output = 0
        i = 0
        for dicom in self:
            mean = dicom.mean
            if mean is None:
                continue
            output = i / (i + 1) * output + mean / (i + 1)
            i += 1
        return output

    @property
    def middle_frame(self) -> DICOM:
        """
        会被修改的属性不应该lazy
        :return:
        """
        if self.middle_frame_uid is not None:
            return self[self.middle_frame_uid]
        else:
            return self[(len(self) - 1) // 2]

    def set_middle_frame(self, instance_uid):
        self.middle_frame_uid = instance_uid

    @property
    def image_positions(self):
        positions = []
        for dicom in self:
            positions.append(dicom.image_position)
        return torch.stack(positions, dim=0)

    @property
    def image_orientations(self):
        orientations = []
        for dicom in self:
            orientations.append(dicom.image_orientation)
        return torch.stack(orientations, dim=0)

    @property
    def unit_normal_vectors(self):
        vectors = []
        for dicom in self:
            vectors.append(dicom.unit_normal_vector)
        return torch.stack(vectors, dim=0)

    @lazy_property
    def series_uid(self):
        study_uid_counter = Counter([d.series_uid for d in self])
        return study_uid_counter.most_common(1)[0][0]

    @lazy_property
    def study_uid(self):
        study_uid_counter = Counter([d.study_uid for d in self])
        return study_uid_counter.most_common(1)[0][0]

    def point_distance(self, coord: torch.Tensor):
        """
        点到序列中每一张图像平面的距离，单位为毫米
        :param coord: 人坐标系坐标，Nx3的矩阵或者长度为3的向量
        :return: 长度为NxM的矩阵或者长度为M的向量，M是序列的长度
        """
        return torch.stack([dicom.point_distance(coord) for dicom in self], dim=1).squeeze()

    def k_nearest(self, coord: torch.Tensor, k, max_dist) -> List[List[DICOM]]:
        """

        :param coord: 人坐标系坐标，Nx3的矩阵或者长度为3的向量
        :param k:
        :param max_dist: 如果距离大于max dist，那么返回一个None
        :return:
        """
        distance = self.point_distance(coord)
        indices = torch.argsort(distance, dim=-1)
        if len(indices.shape) == 1:
            return [[self[i] if distance[i] < max_dist else None for i in indices[:k]]]
        else:
            return [[self[i] if row_d[i] < max_dist else None for i in row[:k]]
                    for row, row_d in zip(indices, distance)]

