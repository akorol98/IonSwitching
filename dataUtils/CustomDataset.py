
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
        columns = [0,1,2,3,4]
        input_list = np.empty((0, self.size_w+1))
        
        if idx-n_before < 0:
            n_to_concat = n_before - idx
            for i in columns:
                input = self.data[:idx+1+n_after, i]
                input = np.concatenate(
                    (self.data[:n_to_concat, i], input),
                    axis=0
                )
                input = input.reshape(1, len(input))
                input_list = np.append(input_list, input, axis=0)
        elif idx+n_after > len(self.data)-1:
            n_to_concat = n_after - (len(self.data)-idx)
            for i in columns:
                input = self.data[idx-n_before:, i]
                input = np.concatenate(
                    (input, self.data[-n_to_concat-1:, i]),
                    axis=0
                )
                input = input.reshape(1, len(input))
                input_list = np.append(input_list, input, axis=0)
        else:
            for i in columns:
                input = self.data[np.r_[idx-n_before:idx, idx, idx+1:idx+1+n_after], i]
                input = input.reshape(1, len(input))
                input_list = np.append(input_list, input, axis=0)
            
#         if idx-n_before < 0:
#             n_to_concat = n_before - idx
#             signal = self.data[np.r_[0:idx, idx+1:idx+1+n_after], 2]
#             signal = np.concatenate(
#                 (self.data[:n_to_concat, 2], signal),
#                 axis=0
#             )
#         elif idx+n_after > len(self.data)-1:
#             n_to_concat = n_after - (len(self.data)-idx)
#             signal = self.data[np.r_[idx-n_before:idx, idx+1:len(self.data)], 2] 
#             signal = np.concatenate(
#                 (signal, self.data[-n_to_concat-1:, 2]),
#                 axis=0
#             )
#         else:
#             signal = self.data[np.r_[idx-n_before:idx, idx+1:idx+1+n_after], 2]
            
#         signal = 2*(signal-signal.min())/(signal.max()-signal.min())-1
        
#         p0 = p0.reshape(1, len(p0))
#         signal = signal.reshape(1, len(signal))
#         input = np.concatenate((signal,p0), axis=0)
    
        if self.train:
            n_open_channels = int(self.data[idx, -1])
            open_channels = torch.tensor(np.zeros(11))
            open_channels[n_open_channels] = 1
            
            sample = {'input': input_list,
                      'open_channels': open_channels
                     }
        else: sample = {'input': input_list}

        if self.transform:
            sample = self.transform(sample)

        return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        if len(sample) > 1:
            input = sample['input']
            open_chennels = sample['open_channels']
            input = torch.from_numpy(input)
            open_channels = torch.from_numpy(open_channels)
            return {'input': input,
                    'open_channels': open_channels
                   }
        else:
            signal = sample['signal']
            signal = torch.from_numpy(signal)
            return {'signal': signal}