import math
import random
from statistics import mean

import torch

from common.utils.constants import UNBC_DATASET_PATH
from common.utils.random_utils import RandomSequence
from databases.unbc import unbc_mcmaster
from databases.unbc.unbc_mcmaster import UNBCMcMasterDataset


class UNBCRNNDataset(UNBCMcMasterDataset):

    def __init__(self, dataset_path=UNBC_DATASET_PATH, excluded_subjects=None, transform=None, init_transform=None,
                 seq_transform=None, sequence_length=16, crop_size=224, apply_balancing=True,
                 apply_balancing_last_frame=False,
                 exclude_black_frames=False, per_category_balancing=False, **kwargs) -> None:
        """
        The Dataset class for getting sequence data from UNBC McMaster database. This class is supposed to be used
        for RNN training and validating process.
        @param dataset_path: the location of the UNBC zip files
        @param excluded_subjects: see cleanup_excluded_subjects method
        @param transform: some preprocessing transformations
        """
        super().__init__(dataset_path, excluded_subjects, transform, init_transform, crop_size,
                         exclude_black_frames, **kwargs)
        self.cleanup_excluded_subjects(self.excluded_subjects)
        self.sequences = self.extract_sequences(sequence_length)
        self.seq_transform = seq_transform
        if apply_balancing:
            self.sequences = self.balance_sequences(self.sequences, apply_balancing_last_frame)
        if per_category_balancing:
            self.sequences = self.per_category_balancing(self.sequences, majority_label=0.0)

    def cleanup_excluded_subjects(self, excluded_subjects: []):
        """
        For better performance in terms of generalisation, we will exclude some subject(s) out of our dataset
        i.e we can exclude one subject for training and exclude (n - 1) subjects for validation (leave one subject out)
        """
        if not excluded_subjects:
            return
        sequences_mapping = unbc_mcmaster.subject_sequence_mapping(self.ids)
        to_delete = []
        for subject in list(sequences_mapping):
            if subject not in excluded_subjects:
                continue
            for sequence in sequences_mapping[subject]:
                to_delete += sequences_mapping[subject][sequence]
            del sequences_mapping[subject]
        for index in sorted(to_delete, reverse=True):
            del self.ids[index]

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        images = []
        all_landmarks = []
        scores = []
        ids = []
        sequence = self.sequences[idx]
        for i in range(sequence[0], sequence[1]):
            sample = self.get_frame_data(i)
            images.append(sample['image'])
            all_landmarks.append(sample['landmarks'])
            scores.append(sample['score'])
            ids.append(sample['id'])

        res = {
            'image': images,
            'landmarks': all_landmarks,
            'score': scores,
            'id': tuple(ids)
        }
        if self.seq_transform:
            res = self.seq_transform(res)
        res = {
            'image': torch.stack(res['image'], dim=0),
            'landmarks': torch.stack(res['landmarks'], dim=0),
            'score': torch.stack(res['score'], dim=0),
            'id': res['id']
        }
        return res

    def extract_sequences(self, sequence_length):
        sequences_mapping = unbc_mcmaster.subject_sequence_mapping(self.ids)
        sequences = []
        for subject in list(sequences_mapping):
            for sequence in sequences_mapping[subject]:
                for i, frame_idx in enumerate(sequences_mapping[subject][sequence]):
                    if i <= len(sequences_mapping[subject][sequence]) - sequence_length:
                        sequences.append((frame_idx, frame_idx + sequence_length))
        return sequences

    def balance_sequences(self, sequences, last_frame=False):
        """
        Since UNBC is a skewed database, with a huge amount of frames belongs to no-pain category
        To balance it, we apply "Random under-sampling" approach, eliminate randomly the sample in majority category
        Reference: L. A. Jeni, J. F. Cohn, and F. De la Torre,
                    “Facing imbalanced data–recommendations for the use of performance metrics,”
        This function will balance the number of sequences has pain and number of sequences has no pain in its sequence
        """
        total_has_pain = 0
        pain_frames = []

        # Firstly, calculate total how many sequences has pain
        for idx, (start, end) in enumerate(sequences):
            sequence_has_pain = False
            for i in range(start, end):
                if last_frame and self.ids[end - 1]['score'] == 0.0:
                    sequence_has_pain = False
                    break
                if self.ids[i]['score'] > 0:
                    sequence_has_pain = True
                    break
            if not sequence_has_pain:
                pain_frames.append(idx)
            total_has_pain += sequence_has_pain

        # Then, delete the no-pain sequences till their numbers are equals
        count = len(sequences) - 2 * total_has_pain
        to_delete = random.sample(pain_frames, count)

        for index in sorted(to_delete, reverse=True):
            del sequences[index]

        return sequences

    def per_category_balancing(self, sequences, majority_label=0.0):
        label_mapping = self.get_sequence_mapping(sequences)
        mean_category = int(mean([len(label_mapping[x]) for x in label_mapping if x != majority_label]))
        for category in label_mapping:
            if category == majority_label:
                continue
            if len(label_mapping[category]) >= mean_category:
                continue
            ctg = label_mapping[category]
            to_duplicate = mean_category - len(ctg)
            idxs = [ctg[random.randrange(0, len(ctg))] for _ in range(to_duplicate)]
            for idx in idxs:
                sequences.append(sequences[idx])
        return sequences
