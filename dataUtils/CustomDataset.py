
import torch
import numpy as np
from torch.utils.data import Dataset
import pandas as pd
# import torchvision.transforms as transforms

class SequensDataset2(Dataset):

    def __init__(self, csv_file, seqL, probThr, num_classes, batch_ixs, transform=None,train=True):
        """
        Args:
            csv_file (string): Path to the csv file with data.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.batch_ixs = batch_ixs
        df = pd.read_csv(csv_file)
        self.data = df.loc[self.batch_ixs].values
        self.indices = self.data[:,-1].astype(int)
        self.seqL = seqL
        self.probThr = probThr
        self.num_classes = num_classes
        self.train=train
        self.transform = transform

    def __len__(self):
        return len(self.data)-2*(self.seqL-1)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()


        
        #seq 1
        seq1 = np.array(self.data[idx:idx+self.seqL-1, -12:-12+self.num_classes])
        seq1_1 = np.array(self.data[    idx+self.seqL-1, -12:-12+self.num_classes])
        
        open_channels = np.array(self.data[    idx+self.seqL-1, 2])
        
        label=np.array(seq1_1)
        label[label>self.probThr]=10
        if(label.sum()>9):
            label=np.argmax(label)
        else:
            label=-1
        
        
        seq1[seq1>self.probThr]=-10
        for i in range(2):
            rows = np.where(seq1.sum(axis=1)>-7)
            columns = np.argmax(seq1[rows], axis=1)
            seq1[rows,columns]=-5
        seq1[seq1>0]=0
        seq1[seq1<0]=1

        seq1_1[np.argmax(seq1_1)]=-1
        seq1_1[np.argmax(seq1_1)]=-1
        seq1_1[seq1_1>0]=0
        seq1_1[seq1_1<0]=1
        
        seq1=np.concatenate((seq1,seq1_1.reshape(1,-1)),axis=0)
        
        #seq 2
        seq2 = np.array(self.data[idx+2*self.seqL-2:idx+self.seqL-1:-1, -12:-12+self.num_classes])
        
        seq2[seq2>self.probThr]=-10
        for i in range(2):
            rows = np.where(seq2.sum(axis=1)>-7)
            columns = np.argmax(seq2[rows], axis=1)
            seq2[rows,columns]=-5
        seq2[seq2>0]=0
        seq2[seq2<0]=1
        
        #seq2=np.concatenate((seq2,seq1_1.reshape(1,-1)),axis=0)
        
        

        ## seq1 and seq2
        rows = np.where(seq1.sum(axis=1)>1.5)
        seq1[rows]=seq1[rows]*0.5
        #rows = np.where(seq2.sum(axis=1)>1.5)
        #seq2[rows]=seq2[rows]*0.5
        
        seq2 = seq2[::-1]
        seq = np.concatenate((seq1, seq2), axis=0).T
        
        
        index=self.indices[idx+self.seqL-1]
        
        
        signal = self.data[idx:idx+2*self.seqL-1, 1]
        
        if self.train:
            
            sample = {'seq': seq,
                      'index': index,
                      'label': label,
                      'lable_true' : open_channels,
                      'signal': signal
                     }
        else: sample = {'seq': seq,
                        'index': index,
                        'label': label
                        
                         }

        if self.transform:
            sample = self.transform(sample)

        return sample


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
#         columns = [1,2,3,4,5,6,7,8,9,10,11,12]
        columns = [1]
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
            n_open_channels = int(self.data[idx, -2])
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

     