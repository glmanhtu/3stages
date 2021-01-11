import glob
import os
import re

import cv2
import scipy.io
import torch
from torch.utils.data import Dataset

DISFA_AUS = ['au_1', 'au_2', 'au_4', 'au_5', 'au_6', 'au_9', 'au_12', 'au_15', 'au_17', 'au_20', 'au_25', 'au_26']


class DISFADataset(Dataset):

    def __init__(self, dataset_path: str, excluded_subjects=None, transforms=None, default_transform=None,
                 crop_size=224):
        self.image_path = os.path.join(dataset_path, "Images", "Left")
        assert os.path.isdir(self.image_path)
        self.frame_labels_path = os.path.join(dataset_path, "ActionUnit_Labels")
        assert os.path.isdir(self.frame_labels_path)
        self.aam_landmarks_path = os.path.join(dataset_path, "Landmark_Points")
        assert os.path.isdir(self.aam_landmarks_path)
        self.transform = transforms
        self.excluded_subjects = excluded_subjects
        self.default_transforms = default_transform

        self.ids = getting_ids(self.frame_labels_path)

    def __enter__(self):
        return self

    def get_frame_data(self, idx):
        item = self.ids[idx]
        subject, frame = item['subject'], item['frame']
        file_path = os.path.join(self.image_path, 'LeftVideo%s_comp.avi' % subject, 'frame-%d.jpg' % frame)
        if not os.path.exists(file_path):
            raise Exception('file %s not exists' % file_path)
        image = cv2.imread(file_path, cv2.IMREAD_COLOR)

        # From Original
        landmarks_file = os.path.join(self.aam_landmarks_path, subject, 'tmp_frame_lm',
                                      "{}_{:04d}_lm.mat".format(subject, frame - 1))
        if not os.path.isfile(landmarks_file):
            landmarks_file = os.path.join(self.aam_landmarks_path, subject, 'tmp_frame_lm',
                                          "l0{:04d}_lm.mat".format(frame))
        landmarks = scipy.io.loadmat(landmarks_file)['pts']

        # landmarks_file = os.path.join(self.dlib_landmarks_path, subject, '%s.npy' % item['id'])
        # landmarks = np.load(landmarks_file).astype(np.float32)

        sample = {
            'landmarks': landmarks,
            'image': image[..., ::-1].copy(),   # from BGR to RGB
            **item
        }
        sample = self.default_transforms(sample)
        for au in DISFA_AUS:
            sample[au] = torch.tensor(sample[au])
        if self.transform:
            sample['image'] = self.transform(sample['image'])
        sample['landmarks'] = torch.from_numpy(sample['landmarks'])
        return sample

    def __len__(self):
        raise NotImplementedError()

    def __getitem__(self, idx):
        raise NotImplementedError()


def getting_ids(frame_labels_path):
    frame_mapping = {}
    frame_labels_pattern = os.path.join(frame_labels_path, '**', '*.txt')
    list_files = glob.glob(frame_labels_pattern, recursive=True)
    for file in list_files:
        subject = os.path.basename(os.path.dirname(file))
        au_name = re.match(r'%s_au(\d+)\.txt' % subject, os.path.basename(file))
        au_name = int(au_name.group(1))
        with open(file) as f:
            for frame in f:
                au_val = frame.strip().split(',')
                frame_id = subject + '_' + au_val[0]
                if frame_id not in frame_mapping:
                    frame_mapping[frame_id] = {}
                frame_mapping[frame_id]['au_%d' % au_name] = float(au_val[1])
    result = []
    for frame_id in frame_mapping:
        subject, frame = tuple(frame_id.split('_'))
        result.append({'id': frame_id, 'subject': subject, 'frame': int(frame), **frame_mapping[frame_id]})
    result = sorted(result, key=lambda i: (i['subject'], i['frame']))
    return result


def getting_subjects(dataset_path):
    files = os.listdir(os.path.join(dataset_path, 'ActionUnit_Labels'))
    subjects = [x for x in files if os.path.isdir(os.path.join(dataset_path, 'ActionUnit_Labels', x))]
    return sorted(subjects)
