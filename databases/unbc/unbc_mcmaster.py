import glob
import os
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset

from preprocessing.image import CentralCrop, GPAAlignment
from utils.constants import UNBC_DATASET_PATH

black_frames = [
    os.sep.join(['095-tv095', 'tv095t2aeunaff', 'tv095t2aeunaff001']),
    os.sep.join(['095-tv095', 'tv095t2aeunaff', 'tv095t2aeunaff002']),
    os.sep.join(['095-tv095', 'tv095t2aeunaff', 'tv095t2aeunaff003']),
    os.sep.join(['095-tv095', 'tv095t2aeunaff', 'tv095t2aeunaff004']),
    os.sep.join(['095-tv095', 'tv095t2aeunaff', 'tv095t2aeunaff005']),
    os.sep.join(['095-tv095', 'tv095t2aeunaff', 'tv095t2aeunaff006']),
    os.sep.join(['095-tv095', 'tv095t2aeunaff', 'tv095t2aeunaff007'])
]

UNBC_AUS = ['au_4', 'au_6', 'au_7', 'au_9', 'au_10', 'au_12', 'au_20', 'au_25', 'au_26', 'au_43']


class UNBCMcMasterDataset(Dataset):

    def __init__(self, dataset_path: str, excluded_subjects=None, transforms=None, default_transform=None,
                 crop_size=224, exclude_black_frames=False):
        self.dataset_path = dataset_path
        self.image_path = os.path.join(dataset_path, "Images")
        assert os.path.isdir(self.image_path)
        self.frame_labels_path = os.path.join(dataset_path, "Frame_Labels")
        assert os.path.isdir(self.frame_labels_path)
        self.aam_landmarks_path = os.path.join(dataset_path, "AAM_landmarks")
        assert os.path.isdir(self.aam_landmarks_path)
        self.sequence_labels_path = os.path.join(dataset_path, "Sequence_Labels")
        self.dlib_landmarks = os.path.join(dataset_path, "dlib_landmarks")
        self.transform = transforms
        self.excluded_subjects = excluded_subjects

        self._default_transforms = torchvision.transforms.Compose([])
        if default_transform is not None:
            self._default_transforms = default_transform
        else:
            self._default_transforms = torchvision.transforms.Compose([GPAAlignment(), CentralCrop(crop_size)])

        self.ids = getting_ids(self.dataset_path, exclude_black_frames)

    def __enter__(self):
        return self

    def get_frame_data(self, idx):
        item = self.ids[idx]
        frame_id = item['id']
        image = read_image(self.dataset_path, frame_id)
        landmarks = read_landmarks(self.dataset_path, frame_id)
        # landmarks_file = os.path.join(self.dlib_landmarks, '%s.npy' % item['id'])
        # landmarks = np.load(landmarks_file).astype(np.float32)
        sample = {
            'landmarks': landmarks,
            'image': image[..., ::-1].copy(),   # from BGR to RGB
            **item
        }
        sample = self._default_transforms(sample)
        for au in UNBC_AUS:
            sample[au] = torch.tensor(sample[au])
        sample['score'] = torch.tensor(sample['score'])
        if self.transform:
            sample['image'] = self.transform(sample['image'])
        sample['landmarks'] = torch.from_numpy(sample['landmarks'])
        return sample

    def get_sequence_mapping(self, sequences):
        label_mapping = {}
        for idx, (start, end) in enumerate(sequences):
            label = self.ids[end - 1]['score']
            if label not in label_mapping:
                label_mapping[label] = []
            label_mapping[label].append(idx)
        return label_mapping

    def __len__(self):
        raise NotImplementedError()

    def __getitem__(self, idx):
        raise NotImplementedError()


def read_image(dataset_path: str, frame_id):
    image_path = os.path.join(dataset_path, "Images")
    file_path = os.path.join(image_path, '%s.png' % frame_id)
    return cv2.imread(file_path, cv2.IMREAD_COLOR)


def read_landmarks(dataset_path: str, frame_id):
    landmarks_path = os.path.join(dataset_path, "AAM_landmarks")
    landmarks_file = os.path.join(landmarks_path, '%s_aam.txt' % frame_id)
    with open(landmarks_file) as f:
        landmarks = f.read().split()
    return np.array([float(x) for x in landmarks]).reshape((-1, 2))


def getting_aus(dataset_path: str, frame_id):
    facs_label_path = os.path.join(dataset_path, "Frame_Labels")

    au_files = os.path.join(facs_label_path, 'FACS', '%s_facs.txt' % frame_id)
    aus = {}
    for au in UNBC_AUS:
        aus[au] = 0.
    with open(au_files) as f:
        for line in f:
            au = [int(float(x)) for x in line.split()]
            key = 'au_%d' % au[0]
            if key in aus:
                aus[key] = float(au[1])
            if key == 'au_43':
                # For AU 43, there is no intensity. If it appear, then the intensity is 1
                aus[key] = 1.
    return aus


def getting_ids(dataset_path, exclude_black_frames):
    """
    Read content of the frame labels zip file to get the list of ids and its scores
    id will have the format of <subject_id>/<sequence_id>/<file_name>
    Score will be PSPI score
    @return: list
    """
    result = []
    frame_labels_path = os.path.join(dataset_path, "Frame_Labels")
    frame_labels_pattern = os.path.join(frame_labels_path, 'PSPI', '**', '*_facs.txt')
    list_files = glob.glob(frame_labels_pattern, recursive=True)
    for file in list_files:
        frame_id = os.path.join(*Path(file).parts[-3:]).replace('_facs.txt', '')
        if exclude_black_frames and frame_id in black_frames:
            continue
        with open(file) as f:
            file_content = f.read().strip()
        aus = getting_aus(dataset_path, frame_id)
        subject, sequence, frame = frame_id.split(os.sep)
        frame = {'id': frame_id, 'subject': subject, 'score': float(file_content), **aus}
        result.append(frame)
    result = sorted(result, key=lambda i: i['id'])
    return result


def getting_score_mapping(ids):
    result = {}
    for idx, item in enumerate(ids):
        if item['score'] not in result:
            result[item['score']] = []
        result[item['score']].append(idx)
    return result


def subject_sequence_mapping(ids):
    result = {}
    for index, item in enumerate(ids):
        subject_id, sequence_id, file_name = item['id'].split(os.sep)
        if subject_id not in result:
            result[subject_id] = {}
        if sequence_id not in result[subject_id]:
            result[subject_id][sequence_id] = []
        result[subject_id][sequence_id].append(index)
    return result


def get_subjects(dataset_path=UNBC_DATASET_PATH):
    """
    Get all subjects from database
    @param dataset_path: /path/to/unbc-database
    @return: list
    """
    result = []
    frame_labels_path = os.path.join(dataset_path, "Frame_Labels")
    assert os.path.isdir(frame_labels_path)
    frame_labels_pattern = os.path.join(frame_labels_path, 'PSPI', '**', '*_facs.txt')
    list_files = glob.glob(frame_labels_pattern, recursive=True)
    for file in list_files:
        frame_id = os.path.join(*Path(file).parts[-3:]).replace('_facs.txt', '')
        subject_id, sequence_id, file_name = frame_id.split(os.sep)
        if subject_id not in result:
            result.append(subject_id)

    return sorted(result)
