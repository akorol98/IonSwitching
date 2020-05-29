
import torch
import torch.nn as nn

class Seq_CNN(nn.Module):
    def __init__(self, ngpu):
        super(Seq_CNN, self).__init__()
        self.ngpu = ngpu

        
        self.main_conv = nn.Sequential(

            nn.Conv1d(11, 64, 5, 1, 0, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),

            nn.Conv1d(64, 64, 5, 1, 0, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True)
        )        
        
        self.fc = nn.Sequential(

            nn.Linear(64*41, 256),
            nn.LeakyReLU(),
             
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            
            nn.Linear(128, 64),
            nn.LeakyReLU(),

            nn.Linear(64, 11)
        )

    def forward(self, input):

        main_conv_out = self.main_conv(input).view(-1, 64*41)
        
        return self.fc(main_conv_out)
#         return main_conv_out
#         return self.fc(input.view(-1, 49*11))

class OpenChannelsClassifier_CNN2(nn.Module):
    def __init__(self, ngpu):
        super(OpenChannelsClassifier_CNN2, self).__init__()
        self.ngpu = ngpu

        
        self.main_conv = nn.Sequential(

            nn.Conv1d(1, 64, 3, 1, 0, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),

            nn.Conv1d(64, 64, 3, 1, 0, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True)
        )        
        
        self.fc = nn.Sequential(

            nn.Linear(64*47, 256),
            nn.LeakyReLU(),
             
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            
            nn.Linear(128, 64),
            nn.LeakyReLU(),

            nn.Linear(64, 11)
        )

    def forward(self, input):

        main_conv_out = self.main_conv(input).view(-1, 64*47)
        
        return self.fc(main_conv_out)
#         return main_conv_out


class OpenChannelsClassifier_CNN(nn.Module):
    def __init__(self, ngpu):
        super(OpenChannelsClassifier_CNN, self).__init__()
        self.ngpu = ngpu

        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 16, 1, 1, 0, bias=False),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv1d(1, 16, 3, 1, 0, bias=False),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True)
        )
        
        self.conv5 = nn.Sequential(
            nn.Conv1d(1, 16, 5, 1, 0, bias=False),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True)
        )
        
        self.conv7 = nn.Sequential(
            nn.Conv1d(1, 16, 7, 1, 0, bias=False),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True)
        )
        
        self.main_conv = nn.Sequential(

            nn.Conv1d(16, 64, 5, 1, 0, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),

            nn.Conv1d(64, 64, 5, 1, 0, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True)

        )        
        
        self.fc = nn.Sequential(

            nn.Linear(64*184, 2048),
            nn.LeakyReLU(),
            
            nn.Dropout(0.5),
            
            nn.Linear(2048, 128),
            nn.LeakyReLU(),
            
            nn.Linear(128, 64),
            nn.LeakyReLU(),

            nn.Linear(64, 11)
        )

    def forward(self, input):
        conv_out1 = self.conv1(input)
        conv_out3 = self.conv3(input)
        conv_out5 = self.conv5(input)
        conv_out7 = self.conv7(input)
        conv_out = torch.cat([conv_out1, conv_out3, conv_out5, conv_out7], dim=2)
        main_conv_out = self.main_conv(conv_out).view(-1, 64*184)
        
        return self.fc(main_conv_out)
#         return main_conv_out


class OpenChannelsClassifier_FC(nn.Module):
    def __init__(self, ngpu):
        super(OpenChannelsClassifier_FC, self).__init__()
        self.ngpu = ngpu
        
        self.fc = nn.Sequential(

            nn.Linear(1, 32),
            nn.LeakyReLU(),
            
            nn.Linear(32, 64),
            nn.LeakyReLU(),
            
            nn.Linear(64, 128),
            nn.LeakyReLU(),
            
            nn.Linear(128, 11)
        )
        
    def forward(self, input):
        return self.fc(input)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
