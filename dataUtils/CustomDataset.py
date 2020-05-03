
import torch
import numpy as np
from torch.utils.data import Dataset
import pandas as pd
# import torchvision.transforms as transforms


class IonSwitchingDataset(Dataset):
    """Ion Switching Dataset."""

    def __init__(self, csv_file, window_size=1000, slice_ratio=0.5, concat_value=0.0,
                 train=True, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with data.
            windowsize (int): Size of input data for one event.
            slice_ratio (double): (amount data bifore observed point)/window_size.
            concat_value (double): Value taht concatenates to edges.
            train (boolean): Data for train or prediction, defalt train (True).
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        self.data = pd.read_csv(csv_file).values
        self.size_w = window_size
        self.slice_ratio = slice_ratio
        self.transform = transform
        self.concat_value = concat_value
        self.train = train

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        n_before = int(self.size_w*self.slice_ratio)
        n_after = self.size_w-n_before
        

        if idx-n_before < 0:
            n_to_concat = n_before - idx
            signal = self.data[:idx+1+n_after, 1]
            signal = np.concatenate(
                (self.data[:n_to_concat, 1], signal),
                axis=0
            )
        elif idx+n_after > len(self.data)-1:
            n_to_concat = n_after - (len(self.data)-idx)
            signal = self.data[idx-n_before:, 1]
            if n_to_concat > 0:
                signal = np.concatenate(
                    (signal, self.data[-n_to_concat-1:, 1]),
                    axis=0
                )
        else:
            signal = self.data[np.r_[idx-n_before:idx, idx, idx+1:idx+1+n_after], 1]
            
#         signal = 2*(signal-signal.min())/(signal.max()-signal.min())-1
        
        if self.train:
            n_open_channels = int(self.data[idx, 2])
            open_channels = torch.tensor(np.zeros(11))
            open_channels[n_open_channels] = 1

            sample = {'signal': signal,
                      'open_channels': open_channels}
        else: sample = {'signal': signal}

        if self.transform:
            sample = self.transform(sample)

        return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        if len(sample) > 1:
            signal, open_channels = sample['signal'], sample['open_channels']
            signal = torch.from_numpy(signal)
            open_channels = torch.from_numpy(open_channels)
            return {'signal': signal, 'open_channels': open_channels}
        else:
            signal = sample['signal']
            signal = torch.from_numpy(signal)
            return {'signal': signal}