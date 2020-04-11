
import torch
import numpy as np
from torch.utils.data import Dataset
import pandas as pd
# import torchvision.transforms as transforms


class IonSwitchingDataset(Dataset):
    """Ion Switching Dataset."""

    def __init__(self, csv_file, window_size=1000, slice_ratio=0.5, concat_value=0.0,
                 transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with data.
            windowsize (int): Size of input data for one event.
            slice_ratio (double): (amount data bifore observed point)/window_size
            concat_value (double): Value taht concatenates to edges
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        self.data = pd.read_csv(csv_file)
        self.size_w = window_size
        self.slice_ratio = slice_ratio
        self.transform = transform
        self.concat_value = concat_value

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        n_before = int(self.size_w*self.slice_ratio)
        n_after = self.size_w-n_before

        if idx-n_before < 0:
            n_to_concat = n_before - idx
            signal = self.data.signal[:idx+n_after].values
            signal = np.concatenate(
                (np.ones(n_to_concat)*self.concat_value, signal),
                axis=0
            )
        elif idx+n_after > len(self.data)-1:
            n_to_concat = n_after - (len(self.data)-idx-1)
            signal = self.data.signal[idx:].values
            signal = np.concatenate(
                (signal, np.ones(n_to_concat)*self.concat_value),
                axis=0
            )
        else:
            signal = self.data.signal[idx-n_before:idx+n_after].values

        open_channels = self.data.open_channels[idx]

        sample = {'signal': signal,
                  'open_channels': open_channels}

        if self.transform:
            sample = self.transform(sample)

        return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        signal, open_channels = sample['signal'], sample['open_channels']

        signal = torch.from_numpy(signal).float()
        open_channels = torch.from_numpy(open_channels).flat()


#         in_transform = transforms.Compose([transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                                                 std=[0.229, 0.224, 0.225])])
#         signal = in_transform(signal)

        return {'signal': signal, 'open_channels': open_channels}
