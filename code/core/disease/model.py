from copy import deepcopy
from typing import Tuple
import torch
import torchvision.transforms.functional as tf
from ..structure import Study
from ..key_point import KeyPointModel
from ..data_utils import SPINAL_VERTEBRA_ID, SPINAL_VERTEBRA_DISEASE_ID, SPINAL_DISC_ID, SPINAL_DISC_DISEASE_ID


VERTEBRA_POINT_INT2STR = {v: k for k, v in SPINAL_VERTEBRA_ID.items()}
VERTEBRA_DISEASE_INT2STR = {v: k for k, v in SPINAL_VERTEBRA_DISEASE_ID.items()}
DISC_POINT_INT2STR = {v: k for k, v in SPINAL_DISC_ID.items()}
DISC_DISEASE_INT2STR = {v: k for k, v in SPINAL_DISC_DISEASE_ID.items()}


class DiseaseModelBase(torch.nn.Module):
    def __init__(self,
                 kp_model: KeyPointModel,
                 sagittal_size: Tuple[int, int],
                 num_vertebra_diseases=len(SPINAL_VERTEBRA_DISEASE_ID),
                 num_disc_diseases=len(SPINAL_DISC_DISEASE_ID)):
        super().__init__()
        self.sagittal_size = sagittal_size
        self.num_vertebra_diseases = num_vertebra_diseases
        self.num_disc_disease = num_disc_diseases
        self.backbone = deepcopy(kp_model)

    @property
    def out_channels(self):
        return self.backbone.out_channels

    @property
    def num_vertebra_points(self):
        return self.backbone.num_vertebra_points

    @property
    def num_disc_points(self):
        return self.backbone.num_disc_point

    @property
    def kp_parameters(self):
        return self.backbone.kp_parameters

    @property
    def resnet_out_channels(self):
        return self.backbone.resnet_out_channels

    @staticmethod
    def _gen_annotation(study: Study, vertebra_coords, vertebra_scores, disc_coords, disc_scores) -> dict:
        """

        :param study:
        :param vertebra_coords: Nx2
        :param vertebra_scores: V
        :param disc_scores: Dx1
        :return:
        """
        z_index = study.t2_sagittal.instance_uids[study.t2_sagittal_middle_frame.instance_uid]
        point = []
        for i, (coord, score) in enumerate(zip(vertebra_coords, vertebra_scores)):
            vertebra = int(torch.argmax(score, dim=-1).cpu())
            point.append({
                'coord': coord.cpu().int().numpy().tolist(),
                'tag': {
                    'identification': VERTEBRA_POINT_INT2STR[i],
                    'vertebra': VERTEBRA_DISEASE_INT2STR[vertebra]
                },
                'zIndex': z_index
            })
        for i, (coord, score) in enumerate(zip(disc_coords, disc_scores)):
            disc = int(torch.argmax(score, dim=-1).cpu())
            point.append({
                'coord': coord.cpu().int().numpy().tolist(),
                'tag': {
                    'identification': DISC_POINT_INT2STR[i],
                    'disc': DISC_DISEASE_INT2STR[disc]
                },
                'zIndex': z_index
            })
        annotation = {
            'studyUid': study.study_uid,
            'data': [
                {
                    'instanceUid': study.t2_sagittal_middle_frame.instance_uid,
                    'seriesUid': study.t2_sagittal_middle_frame.series_uid,
                    'annotation': [
                        {
                            'data': {
                                'point': point,
                            }
                        }
                    ]
                }
            ]
        }
        return annotation

    def _train(self, sagittals, _, distmaps, v_labels, d_labels, v_masks, d_masks, t_masks) -> tuple:
        masks = torch.cat([v_masks, d_masks], dim=-1)
        return self.backbone(sagittals, distmaps, masks)

    def _inference(self, study: Study, to_dict=False):
        kp_frame = study.t2_sagittal_middle_frame
        # 将图片放缩到模型设定的大小
        sagittal = tf.resize(kp_frame.image, self.sagittal_size)
        sagittal = tf.to_tensor(sagittal).unsqueeze(0)

        v_coord, d_coord, _, feature_maps = self.backbone(sagittal, return_more=True)

        # 将预测的坐标调整到原来的大小，注意要在extract_point_feature之后变换
        height_ratio = self.sagittal_size[0] / kp_frame.size[1]
        width_ratio = self.sagittal_size[1] / kp_frame.size[0]
        ratio = torch.tensor([width_ratio, height_ratio], device=v_coord.device)
        v_coord = (v_coord.float() / ratio).round()[0]
        d_coord = (d_coord.float() / ratio).round()[0]

        v_score = torch.zeros(v_coord.shape[0], self.num_vertebra_diseases)
        v_score[:, 1] = 1

        d_score = torch.zeros(d_coord.shape[0], self.num_disc_disease)
        d_score[:, 0] = 1

        if to_dict:
            return self._gen_annotation(study, v_coord, v_score, d_coord, d_score)
        else:
            return v_coord, v_score, d_coord, d_score

    def forward(self, *args, **kwargs):
        if self.training:
            return self._train(*args, **kwargs)
        else:
            return self._inference(*args, **kwargs)
