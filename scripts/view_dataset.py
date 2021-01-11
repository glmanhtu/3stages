import cv2
import tqdm
from torchvision import transforms
import numpy as np
from databases.disfa.disfa_database_cnn import DISFADatasetCNN
from utils.constants import DISFA_BASE_GPA_LANDMARKS_PATH
from databases.unbc.unbc_mcmaster_cnn import UNBCCNNDataset
from preprocessing.image import GPAAlignment, CentralCrop
import matplotlib.pyplot as plt


# init_transform = transforms.Compose([GPAAlignment(), CentralCrop(160, percent=0.01)])
# dataset = UNBCCNNDataset(dataset_path='/Users/mvu/Documents/Datasets/unbc/dataset', excluded_subjects=[], init_transform=init_transform,
#                          apply_balancing=False, exclude_black_frames=True)

base_lm = np.load(DISFA_BASE_GPA_LANDMARKS_PATH)
init_transform = transforms.Compose([GPAAlignment(base_lm), CentralCrop(160, percent=0.01, close_top=1)])
dataset = DISFADatasetCNN(dataset_path='/home/mvu/Documents/datasets/disfa', excluded_subjects=[], init_transform=init_transform)

def show_landmarks(image, landmarks):
    """Show image with landmarks"""
    plt.figure()
    plt.imshow(image)
    # for idx, landmarks in enumerate(batch_landmarks):
    plt.scatter(landmarks[:, 0], landmarks[:, 1], s=10, marker='.', c='r')
        # plt.annotate(idx, (landmarks[0], landmarks[1]))
    plt.axis('off')
    plt.ioff()
    # plt.pause(0.05)
    # plt.clf()
    plt.show()


images = []
for item in tqdm.tqdm(dataset):
    show_landmarks(item['image'], item['landmarks'])
    # cv2.imshow('frame', item['image'][..., ::-1])
    # cv2.waitKey(1)
