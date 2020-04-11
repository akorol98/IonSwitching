
import torch.nn as nn

class OpenChannelsClassifier(nn.Module):
    def __init__(self, ngpu):
        super(OpenChannelsClassifier, self).__init__()
        self.ngpu = ngpu

        self.fc = nn.Sequential(

            nn.Linear(10000, 8192),
            nn.LeakyReLU(),
            nn.Dropout(0.5),

            nn.Linear(8192, 4096),
            nn.LeakyReLU(),
            nn.Dropout(0.5),

            nn.Linear(4096, 2048),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            
            nn.Linear(2048, 1024),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            
            nn.Linear(1024, 512),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            
            nn.Linear(256, 64),
            nn.LeakyReLU(),
            nn.Dropout(0.5),

            nn.Linear(64, 11)
        )

    def forward(self, signal):
        return self.fc(signal)

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