# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 23:36:41 2026

@author: JDawg
"""

import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import glob
from utils_classical import all_data_to_csv, train_xgb, generate_classical_masks
from utils_deep_learning import train_mask_model
import yaml
from scipy.stats import gaussian_kde
import xgboost as xgb
from sklearn.model_selection import train_test_split
import pandas as pd
import shap

train_dayglow = False
train_masks = True

files = glob.glob(os.path.join(r'D:\gold_l1c_NI\mission_data', '*.nc'))

with open('config.yaml', 'r', encoding='utf-8') as stream:
    species_info_dict = yaml.safe_load(stream)

species_regions = [info['region'] for info in species_info_dict.values()]


if train_dayglow:
    files = [f for f in files if '2020' in f]

    #gathers all emission data from 2020. 
    all_data_to_csv(files
                    , species_regions
                    , out = r'D:\gold_l1c_NI\2020_ni_data.csv'
                    )
    
    
    #dayglow training data
    dayglow_df = pd.read_csv(r'D:\gold_l1c_NI\2020_ni_data.csv')
    dayglow_df =  dayglow_df[(
        (dayglow_df.lat< 60) & 
        (dayglow_df.sza < 110) & 
        (dayglow_df.channel == 'CHA')
        )].reset_index()
    
    
    #trains a dayglow model 
    train_xgb(dayglow_df
              , x_cols = ['ema','sza']
              , y_cols = ['lbh' , '1493', '1356']
              # , save_path = 'cha_dayglow_xgb.json'
              )    


if train_masks:
    files = [f for f in files if '2020' in f]

    # #gathers classical masks for year 2020
    # generate_classical_masks(files, 
    #                              species_regions, 
    #                              dayglow_model_pt = "cha_dayglow_xgb.json", 
    #                              save_path = r'D:\gold_l1c_NI\cha_2020_train',
    #                              channel = 'CHA')

    
    #trains a model that can predict auroral masks in an image.
    ''' I want to train an ensemble of models-- This method is fine.
     I think for preprocessing during storms/active years I need to normalize wrt the 
    scale of the dayglow channels as the activation functions may be sensitive to the actual
    magnitude of the images. Normalizing may not be working for the EIA. Possibly include sza>108 data.
    
    Noise injection along the limb pixels? This could be interesting to increase the possible robustness of the model.
    Specifically doing this for the earlier days of the year may allow the model to generally ignore these values. 
    '''
    weight_dir = 'model_weights_nitrogen_only'
    os.makedirs(weight_dir, exist_ok=True)
    for i in range(10):
        train_mask_model(dataset_path = r'D:\gold_l1c_NI\cha_2020_train',
                           model_save_pt = os.path.join(weight_dir, f'cha_aurora_{i}.pth'),
                           show_plots = True
                           )
        
    

