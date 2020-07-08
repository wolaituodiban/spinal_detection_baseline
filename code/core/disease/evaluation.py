import json
import math
from typing import Dict

import pandas as pd
from tqdm import tqdm

from .model import DiseaseModelBase
from ..data_utils import SPINAL_DISC_DISEASE_ID, SPINAL_VERTEBRA_DISEASE_ID
from ..structure import Study


def distance(coord0, coord1, pixel_spacing):
    x = (coord0[0] - coord1[0]) * pixel_spacing[0]
    y = (coord0[1] - coord1[1]) * pixel_spacing[1]
    output = math.sqrt(x ** 2 + y ** 2)
    return output


def format_annotation(annotations):
    """
    转换直接读取的annotation json文件的格式
    :param annotations:
    :return:
    """
    output = {}
    for annotation in annotations:
        study_uid = annotation['studyUid']
        series_uid = annotation['data'][0]['seriesUid']
        instance_uid = annotation['data'][0]['instanceUid']
        temp = {}
        for point in annotation['data'][0]['annotation'][0]['data']['point']:
            identification = point['tag']['identification']
            coord = point['coord']
            if 'disc' in point['tag']:
                disease = point['tag']['disc']
            else:
                disease = point['tag']['vertebra']
            if disease == '':
                disease = 'v1'
            temp[identification] = {
                'coord': coord,
                'disease': disease,
            }
        output[study_uid] = {
            'seriesUid': series_uid,
            'instanceUid': instance_uid,
            'annotation': temp
        }
    return output


class Evaluator:
    def __init__(self, module: DiseaseModelBase, studies: Dict[str, Study], annotation_path: str, metric='macro f1',
                 max_dist=6, epsilon=1e-5, num_rep=1):
        self.module = module
        self.studies = studies
        with open(annotation_path, 'r') as file:
            annotations = json.load(file)

        self.annotations = format_annotation(annotations)
        self.num_rep = num_rep
        self.metric = metric
        self.max_dist = max_dist
        self.epsilon = epsilon

    def inference(self):
        self.module.eval()
        output = []
        for study in self.studies.values():
            pred = self.module(study, to_dict=True)
            output.append(pred)
        return output

    def confusion_matrix(self, predictions) -> pd.DataFrame:
        """

        :param predictions: 与提交格式完全相同
        :return:
        """
        columns = ['disc_' + k for k in SPINAL_DISC_DISEASE_ID]
        columns += ['vertebra_' + k for k in SPINAL_VERTEBRA_DISEASE_ID]
        output = pd.DataFrame(self.epsilon, columns=columns, index=columns+['wrong', 'not_hit'])

        predictions = format_annotation(predictions)
        for study_uid, annotation in self.annotations.items():
            study = self.studies[study_uid]
            pixel_spacing = study.t2_sagittal_middle_frame.pixel_spacing
            pred_points = predictions[study_uid]['annotation']
            for identification, gt_point in annotation['annotation'].items():
                gt_coord = gt_point['coord']
                gt_disease = gt_point['disease']
                # 确定是椎间盘还是锥体
                if '-' in identification:
                    _type = 'disc_'
                else:
                    _type = 'vertebra_'
                # 遗漏的点记为fn
                if identification not in pred_points:
                    for d in gt_disease.split(','):
                        output.loc['not_hit', _type + d] += 1
                    continue
                # 根据距离判断tp还是fp
                pred_coord = pred_points[identification]['coord']
                pred_disease = pred_points[identification]['disease']
                if distance(gt_coord, pred_coord, pixel_spacing) >= self.max_dist:
                    for d in gt_disease.split(','):
                        output.loc['wrong', _type + d] += 1
                else:
                    for d in gt_disease.split(','):
                        output.loc[_type + pred_disease, _type + d] += 1
        return output

    @staticmethod
    def cal_metrics(confusion_matrix: pd.DataFrame):
        key_point_recall = confusion_matrix.iloc[:-2].sum().sum() / confusion_matrix.sum().sum()
        precision = {col: confusion_matrix.loc[col, col] / confusion_matrix.loc[col].sum() for col in confusion_matrix}
        recall = {col: confusion_matrix.loc[col, col] / confusion_matrix[col].sum() for col in confusion_matrix}
        f1 = {col: 2 * precision[col] * recall[col] / (precision[col] + recall[col]) for col in confusion_matrix}
        macro_f1 = sum(f1.values()) / len(f1)

        # 只考虑预测正确的点
        columns = confusion_matrix.columns
        recall_true_point = {col: confusion_matrix.loc[col, col] / confusion_matrix.loc[columns, col].sum()
                             for col in confusion_matrix}
        f1_true_point = {col: 2 * precision[col] * recall_true_point[col] / (precision[col] + recall_true_point[col])
                         for col in confusion_matrix}
        macro_f1_true_point = sum(f1_true_point.values()) / len(f1)
        output = [('macro f1', macro_f1), ('key point recall', key_point_recall),
                  ('macro f1 (true point)', macro_f1_true_point)]
        output += sorted([(k+' f1 (true point)', v) for k, v in f1_true_point.items()], key=lambda x: x[0])
        return output

    def __call__(self, *args, **kwargs):
        confusion_matrix = None
        for _ in tqdm(range(self.num_rep), ascii=True):
            predictions = self.inference()
            if confusion_matrix is None:
                confusion_matrix = self.confusion_matrix(predictions)
            else:
                confusion_matrix += self.confusion_matrix(predictions)
        output = self.cal_metrics(confusion_matrix)

        i = 0
        while i < len(output) and output[i][0] != self.metric:
            i += 1
        if i < len(output):
            output = [output[i]] + output[:i] + output[i+1:]
        return output
