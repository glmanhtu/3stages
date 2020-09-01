from databases.disfa.disfa_database import DISFADataset, DISFA_AUS


class DISFADatasetCNN(DISFADataset):

    def __init__(self, dataset_path, excluded_subjects=None, transforms=None, init_transform=None,
                 crop_size=224, balancing_aus=None, min_num_aus=1):

        super().__init__(dataset_path, excluded_subjects, transforms, init_transform, crop_size)
        self.min_num_aus = min_num_aus
        if excluded_subjects is not None:
            self.ids = self.clean_excluded_subject(self.ids, excluded_subjects)
        if balancing_aus is not None:
            self.ids = self.apply_balancing(self.ids, balancing_aus)
        self.frame_index_mapping = {}
        for idx, item in enumerate(self.ids):
            self.frame_index_mapping[item['id']] = item

        # aus = {}
        # for sample in self.ids:
        #     for au in DISFA_AUS:
        #         if au not in aus:
        #             aus[au] = {}
        #         if sample[au] not in aus[au]:
        #             aus[au][sample[au]] = 1
        #         else:
        #             aus[au][sample[au]] += 1
        # print('')

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        return self.get_frame_data(idx)

    def clean_excluded_subject(self, ids, excluded_subjects):
        to_delete = []
        for idx, item in enumerate(ids):
            if item['subject'] in excluded_subjects:
                to_delete.append(idx)

        for index in sorted(to_delete, reverse=True):
            del ids[index]
        return ids

    def insert_au(self, index, au, key, val_max=5, val_min=0):
        au[au > val_max] = val_max
        au[au < val_min] = val_min
        for idx, frame_id in enumerate(index):
            self.frame_index_mapping[frame_id][key] = au[idx].item()

    def apply_balancing(self, ids, aus):
        to_remove = []
        to_keep = []
        for idx, item in enumerate(ids):
            total_aus = 0
            for key in aus:
                if key in DISFA_AUS and item[key] > 0:
                    total_aus += 1
            if total_aus >= self.min_num_aus:
                to_keep.append(idx)
            else:
                to_remove.append(idx)

        n_items_to_eliminate = len(to_remove)
        for index in sorted(to_remove, reverse=True):
            if n_items_to_eliminate <= 0:
                break
            del ids[index]
            n_items_to_eliminate -= 1

        return ids

