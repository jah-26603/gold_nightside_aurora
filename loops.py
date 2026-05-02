# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 15:51:07 2026

@author: JDawg
"""



import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import glob
from utils_classical import all_data_to_csv, train_xgb, generate_classical_masks, filled_indices, extract_species_images
from utils_deep_learning import train_mask_model, SimpleUNet
import yaml
from scipy.stats import gaussian_kde
import xgboost as xgb
from sklearn.model_selection import train_test_split
import pandas as pd
import shap
import spaceweather as sw
from tqdm import tqdm
import torch
from sklearn.linear_model import Lasso
from scipy.signal import convolve

#Kp information
kp_df = sw.ap_kp_3h(update = True)
kp_df['datetime'] = kp_df.index
kp_df = kp_df[(kp_df.datetime.dt.year >= 2018) & (kp_df.Ap >= 0)]
kp_df.index = np.arange(len(kp_df))

#file organization 
files = glob.glob(os.path.join(r'D:\gold_l1c_NI\mission_data', '*.nc'))
file_df = pd.DataFrame(data = files, columns = ['filepath'])
dt_list = [f[-29:-15] for f in files]
file_df['datetime'] = pd.to_datetime(dt_list, format = '%Y_%j_%H_%M')
file_df = file_df.sort_values(by = 'datetime')

#filter based on storms
df = pd.merge_asof(file_df, kp_df, on = 'datetime', direction = 'nearest')
df = df[df.Kp >= 5].reset_index()



with open('config.yaml', 'r', encoding='utf-8') as stream:
    species_info_dict = yaml.safe_load(stream)

species_regions = [info['region'] for info in species_info_dict.values()]


channel = 'CHA'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

weight_dir = 'model_weights'
weight_dir = 'model_weights_nitrogen_only'

paths = glob.glob(os.path.join(os.getcwd(),weight_dir, '*.pth'))
models = [
    SimpleUNet().to(device).eval()
    for _ in paths
]

for m, p in zip(models, paths):
    m.load_state_dict(torch.load(p, weights_only=True))


glow_model = xgb.Booster()
glow_model.load_model(r"C:\Users\JDawg\OneDrive\Desktop\England Research\Aurora\night\cha_dayglow_xgb.json")


for f in tqdm(list(df.filepath)):
    if channel not in f:
        continue
    try:
        ds = nc.Dataset(f, 'r')
    except Exception:
        continue
    with ds:
        ema = ds.variables['EMISSION_ANGLE'][:]
        mask = ema < 90

        radiance = ds.variables['RADIANCE'][:]
        wavelength = ds.variables['WAVELENGTH'][:]
        sza = ds.variables['SOLAR_ZENITH_ANGLE'][:]
        lat = ds.variables['REFERENCE_POINT_LAT'][:]
        lon = ds.variables['REFERENCE_POINT_LON'][:]
        datetime = ds.variables['TIME_ET'][:]
        datetime = np.tile(datetime.data, (lat.shape[0], 1))  # (100, 39)
        
        _, one_pixel = filled_indices(wavelength)  # acceptable indices for analysis
        if one_pixel is None:
            continue

        pixel_wavelengths = np.asarray(one_pixel)
        img = np.stack(extract_species_images(radiance, pixel_wavelengths, species_regions)).transpose(1,2,0)
    

        #MODEL EVAL MASKS -> WHERE TO EVAL THE DAYGLOW MODEL
        valid_mask = (~np.isnan(lat) & (ema < 90) & (sza < 108))
        if (valid_mask == 0).all():
            continue
        
        if np.nanmedian(lat.data) < 0:
            continue
        
        #DAYGLOW MODEL PREDICTS
        inp_data = np.stack([ema[valid_mask].data, sza[valid_mask].data]).T
        inp_data = xgb.DMatrix(inp_data, feature_names=['ema', 'sza'])
        pred = glow_model.predict(inp_data)
        pred = np.clip(pred, 0, np.inf)
        pred_full = np.zeros_like(img)
        pred_full[valid_mask] = pred
        pred_full = pred_full
    

        #BORDER OF IMAGE
        bd = np.array([[1,1,1],[1,0,1],[1,1,1]])
        b1 = convolve(ema<90, bd, mode = 'same') < 8
    
        #GATHERS REGULARIZED COEFFICIENTS THROUGH LASSO REGRESSION
        scale_mask = (pred_full[:,:,2] > np.std(pred_full[:,:,2]))
        scale_x = pred_full[scale_mask]
        scale_y = img[scale_mask]
        
        
        
        slopes = np.zeros(scale_x.shape[1])
        biases = np.zeros_like(slopes)
        for c in range(scale_x.shape[1]):
            x = scale_x[:, c].reshape(-1, 1)
            y = scale_y[:, c]
        
            # Optional: remove bad values
            mask = (x[:, 0] > 0) & np.isfinite(x[:, 0]) & np.isfinite(y)
            x = x[mask]
            y = y[mask]
        
            if len(x) > 10:
                mod = Lasso(alpha=0.001, fit_intercept=False)
                mod.fit(x, y)
                slopes[c] = mod.coef_[0]
                biases[c] = mod.intercept_
            else:
                slopes[c] = 1
        
        species = ['lbh', '1493', '1356']
        result = ['raw_scan', 'difference', 'mask']
        
        
        
        #masks per species
        cum_masks = []
        differences = []
        dayglow = []
        for i in range(len(species)):
            dayglow.append((slopes[i]*pred_full[:,:,i] + biases[i]))

            
        dayglow = np.stack(dayglow).transpose(1,2,0)
        raw_scans = (img*(ema<90)[:,:,None])
        
        # geo_info = np.dstack([lat.data, lon.data, sza.data, ema.data])
        # geo_info[np.isnan(geo_info)] = 0 
        # img = np.dstack([geo_info,raw_scans])

        
        # scales = np.nanmax(img, axis = (0,1))[:3]
        # scales = np.hstack([scales, scales])
        # img = img/scales
        img = np.dstack([dayglow, raw_scans])
        img = img/np.nanmax(img, axis = (0,1))
        img = np.clip(img, 0, np.inf)
        img = img[:, :, [0, 1, 3, 4]]

        
        #reform the image to fit inside UNet
        wide_inp_img = np.stack([np.zeros((100,100)) ] * 4).transpose(1,2,0)
        wide_lab_img= np.zeros((100,100))
        R, C, ch = img.shape # row by columns
        start_idx = np.random.randint(100 - C) 
        wide_inp_img[:, start_idx: start_idx + C] = img
        img = wide_inp_img.transpose(2,0,1)
        
        with torch.no_grad():
            y = torch.tensor(img, dtype = torch.float32).to(device)
            pred = torch.sum(torch.stack([torch.sigmoid(m(y.unsqueeze(0))) > .5 for m in models]), dim=0)
        
        plt.figure()
        plt.subplot(1,2,1)
        plt.imshow(np.expm1(y[-1, :, start_idx : start_idx + C].cpu().detach().numpy()), origin = 'lower')
        plt.subplot(1,2,2)
        mask = (pred> 5).squeeze().cpu().detach().numpy()[:,start_idx : start_idx + C]
        mask = mask * (np.sum((raw_scans - dayglow) > 50, axis = -1) > 1)
        plt.imshow(mask, origin = 'lower')
        plt.suptitle(f'{f[-29:-15]}')
        plt.show()
        

