# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 14:57:19 2026

@author: JDawg
"""

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets
from torchvision.transforms import ToTensor
import torch.nn as nn
import os
import glob
import numpy as np
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt



#Dataset
class NightAurora(Dataset):
    
    def __init__(self, file_directory, input_name = 'raw', label_name = 'mask', target_shape = (100,100)):
        
        all_files = glob.glob(os.path.join(file_directory, '*.npy'))
        inp_paths = {}
        label_paths = {}
        
        for f in all_files:
            fname = os.path.basename(f)
            if input_name in fname:
                key = fname.replace(input_name, '')
                inp_paths[key] = f
            elif label_name in fname:
                key = fname.replace(label_name, '')
                label_paths[key] = f
        
        # Match only common keys
        common_keys = sorted(set(inp_paths) & set(label_paths))
        
        self.parent_dir = file_directory
        self.input_paths = [inp_paths[k] for k in common_keys]
        self.label_paths = [label_paths[k] for k in common_keys]
        self.target_shape = target_shape
    def __len__(self):
        return len(self.input_paths)
        
    def __getitem__(self, idx):
        
        input_img = np.load(self.input_paths[idx])
        label_img  = np.load(self.label_paths[idx])

        scales = np.nanmax(input_img, axis = (0,1))
        input_img = input_img/scales
        input_img = input_img[:, :, [0, 1, 3, 4]]

        
        wide_inp_img = np.stack([np.zeros(self.target_shape) ] * 4).transpose(1,2,0)
        wide_lab_img= np.zeros(self.target_shape)
        
        R, C, ch = input_img.shape # row by columns
        start_idx = np.random.randint(self.target_shape[1] - C) 
        
        wide_inp_img[:, start_idx: start_idx + C] = input_img
        wide_lab_img[:, start_idx: start_idx + C] = label_img
        
        wide_inp_img = wide_inp_img.transpose(2,0,1)
        
        return wide_inp_img, wide_lab_img, start_idx, C
        
        

class SimpleUNet(nn.Module):
    def __init__(self, in_ch = 4):
        super().__init__()
        
        # --- Encoder ---
        self.enc1 = self.conv_block(in_ch, 64)
        self.enc2 = self.conv_block(64, 128, pad = 1)
        self.enc3 = self.conv_block(128, 256)
        
        self.pool = nn.MaxPool2d(2)
        
        self.bottleneck = self.conv_block(256, 512)
        
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = self.conv_block(512, 256)
        
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = self.conv_block(256, 128)
        
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = self.conv_block(128, 64)
        
        self.out = nn.Conv2d(64, 1, kernel_size=1)

    def conv_block(self, in_c, out_c, pad = 1, p = .5):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=pad),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=pad),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p)
        )
    
    def forward(self, x):
        
        e1 = self.enc1(x)        
        e2 = self.enc2(self.pool(e1))  
        e3 = self.enc3(self.pool(e2))
        
        b = self.bottleneck(self.pool(e3)) 
        
        d3 = self.up3(b)
        
        d3 = F.pad(d3, (0, np.abs(d3.shape[2] - e3.shape[2]), 0, np.abs(d3.shape[3] - e3.shape[3])))
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        
        return self.out(d1)
    
    


def train_mask_model(dataset_path = r'D:\gold_l1c_NI\cha_2020_train',
                   model_save_pt = 'cha_aurora.pth',
                   show_plots = False):
    
    
    #Dataloader
    dataset = NightAurora(dataset_path)
    train_size = int(0.33 * len(dataset))
    val_size   = int(0.33 * len(dataset))
    test_size  = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        [train_size, val_size, test_size]
    )
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader  = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    #Training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = SimpleUNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)
    criterion = nn.BCEWithLogitsLoss()
    
    epochs = 50
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for x, y, _, _ in loop:
            
            # image/label
            x = x.to(device=device, dtype=torch.float32)
            y = y.to(device=device, dtype=torch.float32)
            
            # loss & back propogation
            optimizer.zero_grad()
            outputs = model(x)              
            loss = criterion(outputs, y.unsqueeze(1))    
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            loop.set_postfix(loss=loss.item())
        
        avg_train_loss = train_loss / len(train_loader)
    
        
        model.eval()
        val_loss = 0.0
        val_loop = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]")
        
        with torch.no_grad():
            for x, y, start_idx, width in val_loop:
                
                #image/label
                x = x.to(device=device, dtype=torch.float32)
                y = y.to(device=device, dtype=torch.float32)
                
                #loss
                outputs = model(x)
                loss = criterion(outputs, y.unsqueeze(1))
                val_loss += loss.item()
                val_loop.set_postfix(loss=loss.item())
        
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
    
    
    loop = tqdm(test_loader)
    model.eval()
    running_loss = 0
    with torch.no_grad():
        for x, y, start_idx, width in loop:
            
            #image/label
            x = torch.tensor(x.clone(), dtype = torch.float32).to(device)
            y = torch.tensor(y, dtype = torch.float32).to(device)
    
            outputs = model(x)
            loss = criterion(outputs, y.unsqueeze(1))    
            running_loss += loss.item()
            loop.set_postfix(loss=loss.item())
    
    
            if show_plots:
                for i in range(len(x)):
        
                    plt.figure()
                    plt.subplot(1,3,1)
                    plt.imshow(x[i,-1,:, start_idx[i]: start_idx[i] + width[i]].detach().cpu().numpy())
                    plt.title('raw scan')
        
                    plt.subplot(1,3,2)
                    plt.imshow(y[i, :, start_idx[i]: start_idx[i] + width[i]].detach().cpu().numpy(), cmap = 'gray')
                    plt.title('classical masks')
        
                    plt.subplot(1,3,3)
                    plt.imshow(torch.sigmoid(outputs[i, -1,:, start_idx[i]: start_idx[i] + width[i]]).detach().cpu().numpy() > .5, cmap = 'gray')
                    plt.title('model predict')
                    plt.show()
                
                
        print(f"Epoch {epoch+1}, Loss: {running_loss / len(val_loader):.4f}")
    
    
    if model_save_pt is not None:
        torch.save(model.state_dict(), model_save_pt)
        print("Auroral Mask model saved.")
