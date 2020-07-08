import json
import os
from collections import OrderedDict
import SimpleITK as sitk


def dicom_metainfo(dicm_path, list_tag):
    """
    获取dicom的元数据信息
    :param dicm_path: dicom文件地址
    :param list_tag: 标记名称列表,比如['0008|0018',]
    :return:
    """
    reader = sitk.ImageFileReader()
    reader.LoadPrivateTagsOn()
    reader.SetFileName(dicm_path)
    reader.ReadImageInformation()
    return [reader.GetMetaData(t) for t in list_tag]


def dicom2array(dcm_path):
    """
    读取dicom文件并把其转化为灰度图(np.array)
    https://simpleitk.readthedocs.io/en/master/link_DicomConvert_docs.html
    :param dcm_path: dicom文件
    :return:
    """
    image_file_reader = sitk.ImageFileReader()
    image_file_reader.SetImageIO('GDCMImageIO')
    image_file_reader.SetFileName(dcm_path)
    image_file_reader.ReadImageInformation()
    image = image_file_reader.Execute()
    if image.GetNumberOfComponentsPerPixel() == 1:
        image = sitk.RescaleIntensity(image, 0, 255)
        if image_file_reader.GetMetaData('0028|0004').strip() == 'MONOCHROME1':
            image = sitk.InvertIntensity(image, maximum=255)
        image = sitk.Cast(image, sitk.sitkUInt8)
    img_x = sitk.GetArrayFromImage(image)[0]
    return img_x


with open(os.path.join(os.path.dirname(__file__), 'static_files/dicom_tag.json'), 'r') as file:
    DICOM_TAG = json.load(file, object_hook=OrderedDict)


def dicom_metainfo_v2(dicm_path: str) -> dict:
    metainfo = dicom_metainfo(dicm_path, DICOM_TAG.values())
    return {k: v for k, v in zip(DICOM_TAG.keys(), metainfo)}


def dicom_metainfo_v3(dicom_path: str) -> (dict, str):
    metainfo = {}
    error_msg = ''
    for k, v in DICOM_TAG.items():
        try:
            temp = dicom_metainfo(dicom_path, [v])[0]

        except RuntimeError as e:
            temp = None
            error_msg += str(e)
        metainfo[k] = temp
    return metainfo, error_msg


def read_one_dcm(dcm_path):
    return dicom_metainfo_v2(dcm_path), dicom2array(dcm_path)
