

# In[1]:


import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from importlib import reload


# In[2]:


def save(model, epoch, loss, path_to_save):
    torch.save({
                'model': model.module.state_dict(),
                'epoch': epoch,
                'loss': loss,
                },
                path_to_save)

workers = 20
btch_sz = 4096
ws = 500
sr = 0.5
ngpu = 4


# In[3]:


cuda = True if torch.cuda.is_available() else False
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


# # Format Dataloaders

# In[4]:


from dataUtils.CustomDataset import IonSwitchingDataset

ion_dataset_train = IonSwitchingDataset('data/train_new.csv', window_size=ws, slice_ratio=sr)
ion_dataset_test = IonSwitchingDataset('data/test_new.csv', window_size=ws, slice_ratio=sr)
ion_dataset_submmit = IonSwitchingDataset('data/test.csv', window_size=ws, slice_ratio=sr, train=False)


# In[5]:


dataloader = DataLoader(ion_dataset_train, batch_size=btch_sz, shuffle=True, num_workers=workers)
dataloader_test = DataLoader(ion_dataset_test, batch_size=25000, shuffle=False, num_workers=workers)
dataloader_submmit = DataLoader(ion_dataset_submmit, batch_size=btch_sz, shuffle=False, num_workers=workers)



# # Load Model

# In[8]:


import models.FCC_2 as FCC
reload(FCC)

model = FCC.OpenChannelsClassifier(1)
model.apply(FCC.weights_init)
print(sum(p.numel() for p in model.parameters()))
model.to(device)

if (device.type == 'cuda') and (ngpu > 1):
    model = nn.DataParallel(model, list(range(ngpu)))
else:
    model = nn.DataParallel(model)


# # Optimirez and Criterion Loss

# In[12]:


# lr = 0.001
lr = 0.03
criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=lr)
optimizer = torch.optim.SGD(model.parameters(), lr=lr)


# In[13]:


train_losses = np.array([])
train_accurasy = np.array([])
train_f1 = np.array([])
test_accurasy = np.array([])
test_f1 = np.array([])


# In[14]:


batch_test = next(iter(dataloader_test))


# # Training Loop

# In[ ]:


from sklearn.metrics import f1_score

num_epochs = 50
monit_res = 5
for epoch in range(num_epochs):

    running_loss = 0.0
    for i, batch in enumerate(dataloader):
        model.train()
        optimizer.zero_grad()
        
        input, labels = batch['input'], batch['open_channels']
        input = input.to(device).float().view(len(input), 12, ws+1)
        labels = labels.to(device).float()
        
#         open_channels = batch['open_channels_gessed']
#         open_channels = open_channels.to(device).float().view(len(open_channels), 1, ws+1)
#         n_gessed = len(open_channels[open_channels == -1])
#         open_channels[open_channels == -1] = torch.cuda.FloatTensor(np.random.choice(10, n_gessed, p=probs))
    
    
        output = model(input)
        
        loss = criterion(output, torch.max(labels, 1)[1])
        loss.backward()
        optimizer.step()        
        
        running_loss += loss.item()
        
        
        if i%monit_res == monit_res-1:
            model.eval()
            predicted_idx = torch.max(output, 1)[1]
            true_idx = torch.max(labels, 1)[1]
            predicted_idx = predicted_idx.cpu().detach()
            true_idx = true_idx.cpu().detach()
            accurasy = sum(true_idx == predicted_idx).item()/len(true_idx)     
            f1 = f1_score(true_idx, predicted_idx, average='macro')
            train_accurasy = np.append(train_accurasy, accurasy)
            train_f1 = np.append(train_f1, f1)
            
            t_accurasy = 0
            t_f1 = 0
            
            input, labels = batch_test['input'], batch_test['open_channels']
            input = input.to(device).float().view(len(batch_test['input']), 12, ws+1)
            labels = labels.to(device).float()

#             open_channels = batch['open_channels_gessed']
#             open_channels = open_channels.to(device).float().view(len(open_channels), 1, ws+1)
#             n_gessed = len(open_channels[open_channels == -1])
#             open_channels[open_channels == -1] = torch.cuda.FloatTensor(np.random.choice(10, n_gessed, p=probs))
            
            with torch.no_grad():
                output = model(input)
            predicted_idx = torch.max(output, 1)[1]
            true_idx = torch.max(labels, 1)[1]
            predicted_idx = predicted_idx.cpu().detach()
            true_idx = true_idx.cpu().detach()
            t_accurasy = sum(true_idx == predicted_idx).item()/len(true_idx)     
            t_f1 = f1_score(true_idx, predicted_idx, average='macro')
            test_accurasy = np.append(test_accurasy, t_accurasy)
            
            test_f1 = np.append(test_f1, t_f1)

            
            train_losses = np.append(train_losses, running_loss/monit_res)
            print('[{}/{}] [{}/{}], loss: {}, accyrasy: {}, f1: {}  {}'.format(epoch,
                                                     num_epochs,
                                                     i,
                                                     len(dataloader),
                                                     round(running_loss/monit_res, 4),
                                                     round(accurasy, 4),
                                                     round(f1, 4),
                                                     round(t_f1, 4)
                                                    )
                 )
            running_loss = 0.0

    save(model=model, epoch=epoch, loss=np.array([train_losses,train_accurasy]), path_to_save='test.pth' )

# In[21]:

plt.figure(figsize=(10,7))
plt.plot(train_losses, label='cost: {}'.format(round(train_losses[-1], 4)))
plt.plot(train_accurasy, label='accurasy: {}'.format(accurasy))
plt.plot(train_f1, label='F1 score: {}'.format(round(train_f1[-1], 4)))
plt.plot(test_accurasy, label='test accurasy: {}'.format(t_accurasy))
plt.plot(test_f1, label='test F1 score: {}'.format(round(test_f1[-1], 4)))
plt.plot([], [], ' ', label="epoch: {},  lr: {},  ws: {},  bs: {}\n\n{}".format(num_epochs, lr, ws, btch_sz, model))
plt.title('test #17', fontsize=15, loc='left')
plt.legend(fontsize=15, bbox_to_anchor=(1, 1))
plt.savefig('test.png')







