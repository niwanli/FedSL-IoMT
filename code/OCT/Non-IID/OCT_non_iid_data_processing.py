import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import random
import numpy as np

# ==============================================================
# Custom dataset prepration in Pytorch format
class SkinData(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        X = Image.open(self.df['path'][index]).resize((64, 64)).convert('RGB')
        y = torch.tensor(int(self.df['target'][index]))

        if self.transform:
            X = self.transform(X)

        return X, y


# dataset_iid() will create a dictionary to collect the indices of the data samples randomly for each client
# IID chest_xray datasets will be created based on this
def dataset_iid(dataset, num_users):
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    all_idxs0 = []
    all_idxs1 = []
    for idx in range(len(dataset)):
        if dataset['target'][idx] == 0:
            all_idxs0.append(idx)
    all_idxs1 = list(set(all_idxs) - set(all_idxs0))
    num_items0 = int(dataset['target'].value_counts()[0] / num_users)
    num_items1 = int(dataset['target'].value_counts()[1] / num_users)
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs0, num_items0, replace=False))
        all_idxs0 = list(set(all_idxs0) - dict_users[i])
        dict_users[i] = dict_users[i] | set(np.random.choice(all_idxs1, num_items1, replace=False))
        all_idxs1 = list(set(all_idxs1) - dict_users[i])
    return dict_users

# Non-IID chest_xray datasets will be created based on this
def dataset_non_iid(dataset, num_users):
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    all_idxs0 = []
    all_idxs1 = []
    for idx in range(len(dataset)):
        if dataset['target'][idx]==0:
            all_idxs0.append(idx)
    all_idxs1 = list(set(all_idxs) - set(all_idxs0))
    num_items0 = int(dataset['target'].value_counts()[0] / num_users)
    num_items1 = int(dataset['target'].value_counts()[1] / num_users)
    for i in range(num_users):
        if i== 0:
            dict_users[i] = set(np.random.choice(all_idxs0, num_items0*0, replace=False))
        elif i== num_users-1:
            dict_users[i] = set(np.random.choice(all_idxs0, num_items0*2, replace=False))
            all_idxs0 = list(set(all_idxs0) - dict_users[i])
        else:
            dict_users[i] = set(np.random.choice(all_idxs0, num_items0, replace=False))
            all_idxs0 = list(set(all_idxs0) - dict_users[i])
    for i in range(num_users):
        if i== 0:
            dict_users[i] = dict_users[i] | set(np.random.choice(all_idxs1, num_items1*2, replace=False))
            all_idxs1 = list(set(all_idxs1) - dict_users[i])
        elif i== num_users-1:
            dict_users[i] = dict_users[i]| set(np.random.choice(all_idxs1, num_items1*0, replace=False))
            all_idxs0 = list(set(all_idxs0) - dict_users[i])
        else:
            dict_users[i] = dict_users[i] | set(np.random.choice(all_idxs1, num_items1, replace=False))
            all_idxs1 = list(set(all_idxs1) - dict_users[i])
    return dict_users

# Half of OCT dataset
def dataset_split(dataset):
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    all_idxs0 = []
    all_idxs1 = []
    all_idxs2 = []
    all_idxs3 = []
    for idx in range(len(dataset)):
        if dataset['target'][idx]==0:
            all_idxs0.append(idx)
        elif dataset['target'][idx]==1:
            all_idxs1.append(idx)
        elif dataset['target'][idx]==2:
            all_idxs2.append(idx)
        else:
            all_idxs3.append(idx)
    num_items0 = int(dataset['target'].value_counts()[0] / 2)
    num_items1 = int(dataset['target'].value_counts()[1] / 2)
    num_items2 = int(dataset['target'].value_counts()[2] / 2)
    num_items3 = int(dataset['target'].value_counts()[3] / 2)
    dict_users = set(np.random.choice(all_idxs0, num_items0, replace=False))
    dict_user = dict_users | set(np.random.choice(all_idxs1, num_items1, replace=False))
    dict_users = dict_users | set(np.random.choice(all_idxs2, num_items2, replace=False))
    dict_users = dict_users | set(np.random.choice(all_idxs3, num_items3, replace=False))
    return dict_users

