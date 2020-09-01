import copy
import math
import random
from statistics import mean

from utils.constants import UNBC_DATASET_PATH
from databases.unbc import unbc_mcmaster
from databases.unbc.unbc_mcmaster import UNBCMcMasterDataset, UNBC_AUS


class UNBCCNNDataset(UNBCMcMasterDataset):

    def __init__(self, dataset_path=UNBC_DATASET_PATH, excluded_subjects: [] = None, transform=None,
                 init_transform=None, crop_size=224, apply_balancing=True, per_category_balancing=False,
                 exclude_black_frames=False, aus_to_balance=None, **kwargs):
        """
        The Dataset class for getting data from UNBC McMaster database. This class is supposed to be used for CNN
        training and validating process.
        @param dataset_path: the location of the UNBC zip files
        @param excluded_subjects: see cleanup_excluded_subjects method
        @param transform: some preprocessing transformations
        """
        super().__init__(dataset_path, excluded_subjects, transform, init_transform, crop_size,
                         exclude_black_frames, **kwargs)

        self.cleanup_excluded_subjects(self.excluded_subjects)
        if apply_balancing:
            self.ids = self.balancing_data(self.ids, majority_class=0.0)
        if per_category_balancing:
            self.ids = self.per_category_balancing(self.ids, 0.0)
        if aus_to_balance is not None:
            self.ids = self.aus_balancing(self.ids, aus_to_balance)

        # aus = {}
        # for sample in self.ids:
        #     for au in UNBC_AUS:
        #         if au not in aus:
        #             aus[au] = {}
        #         if sample[au] not in aus[au]:
        #             aus[au][sample[au]] = 1
        #         else:
        #             aus[au][sample[au]] += 1
        # print('')

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
        for index in sorted(to_delete, reverse=True):
            del self.ids[index]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        return self.get_frame_data(idx)

    def aus_balancing(self, ids, aus):
        no_aus = []
        has_au = []
        for idx, item in enumerate(ids):
            frame_has_au = False
            for key in aus:
                if key in UNBC_AUS and item[key] > 0:
                    frame_has_au = True
                    break
            if frame_has_au:
                has_au.append(idx)
            else:
                no_aus.append(idx)

        n_items_to_eliminate = len(no_aus)
        for index in sorted(no_aus, reverse=True):
            if n_items_to_eliminate <= 0:
                break
            del ids[index]
            n_items_to_eliminate -= 1

        return ids

    def balancing_data(self, ids, majority_class):
        """
        Since UNBC is a skewed database, with a huge amount of frames belongs to no-pain category
        To balance it, we apply "Random under-sampling" approach, eliminate randomly the sample in majority category
        Reference: L. A. Jeni, J. F. Cohn, and F. De la Torre,
                    “Facing imbalanced data–recommendations for the use of performance metrics,”
        """
        categories_mapping = unbc_mcmaster.getting_score_mapping(ids)
        negative_examples = len(categories_mapping[majority_class])
        positive_examples = sum(len(categories_mapping[x]) for x in categories_mapping if x != majority_class)

        n_items_to_eliminate = negative_examples - positive_examples
        random.shuffle(categories_mapping[majority_class])
        items_to_eliminate = categories_mapping[majority_class][:n_items_to_eliminate]
        for index in sorted(items_to_eliminate, reverse=True):
            del ids[index]

        return ids

    def per_category_balancing(self, ids, majority_class):
        categories_mapping = unbc_mcmaster.getting_score_mapping(ids)
        mean_category = int(mean([len(categories_mapping[x]) for x in categories_mapping if x != majority_class]))
        for category in categories_mapping:
            ctg = categories_mapping[category]
            if category == majority_class:
                continue
            if len(ctg) >= mean_category:
                continue
            to_duplicate = mean_category - len(ctg)
            idxs = [ctg[random.randrange(0, len(ctg))] for _ in range(to_duplicate)]
            for idx in idxs:
                ids.append(copy.deepcopy(ids[idx]))

        return ids
