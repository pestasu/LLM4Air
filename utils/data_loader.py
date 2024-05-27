import os
import datetime
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from utils.tools import convert_tsf_to_dataframe
import warnings

warnings.filterwarnings('ignore')


class get_dataset(Dataset):
    def __init__(self, args, mode='train'):
        self.args = args
        self.__read_data__() 

    def __read_data__(self, mode):
        data = np.load(os.path.join(self.args.data_root_dir, self.args.data_dir, mode+'.npz'))
        cols_data = df_raw.columns[1:]
        df_data = df_raw[cols_data]

        self.x = data['x']
        self.y = data['y']
        
    def __getitem__(self, index):
        sample = {
            'index': index,
            'x': torch.Tensor(self.x[index]),
            'y': torch.Tensor(self.y[index])
        } 
        return sample

    def __len__(self):
        return len(self.data_x) - self.args.seq_len - self.args.pred_len + 1


def get_dataloader(args):

    datasets = {
        'train': get_dataset(args, mode='train'),
        'valid': get_dataset(args, mode='valid'),
        'test': get_dataset(args, mode='test')
    }

    scalers = datasets['train'].scaler
    
    dataLoader = {
        ds: DataLoader(datasets[ds],
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    shuffle=True)
        for ds in datasets.keys()
    }
    return dataLoader, scalers