# -*- coding: utf-8 -*-
"""
Created on Mon Jan 26 19:48:44 2026

@author: JDawg
"""


import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
import pandas as pd
import shap
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import netCDF4 as nc
from scipy.signal import convolve
from sklearn.linear_model import Lasso



def all_data_to_csv(files : list, species_regions, out = r'D:\gold_l1c_NI\2020_ni_data.csv'):
    
    '''Just iterates through all of the files and gathers the:
        time, sza, ema, lat, lon, lbh, 1493, 1356, and channel information
        for each pixel
        
        Saves a pandas dataframe of all this tabular information.'''
    
    data = []
    for f in tqdm(files):
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
            img = extract_species_images(radiance, pixel_wavelengths, species_regions)
    
            
            if 'CHA' in f:
                channel = 'CHA'
            else:
                channel = 'CHB'
            data.append([datetime[mask], sza[mask], ema[mask], lat[mask], lon[mask], img[0][mask], img[1][mask], img[2][mask], [channel]*len(lon[mask])])
            
    df = pd.DataFrame(np.hstack(data).T, columns = ['time', 'sza', 'ema', 'lat','lon', 'lbh', '1493', '1356', 'channel'])
    df.to_csv(out)


def extract_species_images(radiance, pixel_wavelengths, species_regions):
    if np.ma.isMaskedArray(radiance):
        radiance_values = radiance.filled(np.nan)
    else:
        radiance_values = radiance

    bound_index_cache = {}
    for regions in species_regions:
        for lb, ub in regions:
            if lb not in bound_index_cache:
                bound_index_cache[lb] = np.abs(pixel_wavelengths - lb).argmin()
            if ub not in bound_index_cache:
                bound_index_cache[ub] = np.abs(pixel_wavelengths - ub).argmin()

    images = []
    for regions in species_regions:
        brightness = np.zeros(radiance_values.shape[:2], dtype=np.float64)
        for lb, ub in regions:
            lb_idx = bound_index_cache[lb]
            ub_idx = bound_index_cache[ub]
            brightness += np.nansum(radiance_values[:, :, lb_idx:ub_idx + 1], axis=-1)
        images.append(0.04 * brightness)

    return images



def filled_indices(wavelength):
    # Create a boolean mask for finite values
    mask = np.isfinite(wavelength)
    
    # Get indices where the mask is True
    filled_indices = np.argwhere(mask.any(axis=2))  # Find non-empty pixel indices
    
    if filled_indices.size == 0:
        # Handle case when there are no filled indices
        return filled_indices, None

    # Extract one pixel's wavelength values
    one_pixel = wavelength[filled_indices[0, 0], filled_indices[0, 1], :]
    
    return filled_indices, one_pixel




def train_xgb(df, x_cols, y_cols, save_path : str = None ):
    
    '''
    df: dataframe
    x_cols: input variable columns
    y_cols: output prediction
    save_path: if not None, will save the model
    '''
    
    X = df[x_cols].astype(float)
    y = df[y_cols].astype(float)


    X_train, X_test, y_train, y_test = train_test_split(X.to_numpy(), y.to_numpy(), test_size=0.2, random_state = 42)
    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=list(X.columns))
    dvalid = xgb.DMatrix(X_test, label=y_test, feature_names=list(X.columns))


    param = {
        'subsample': 1,
        "max_depth": 4,
        'reg_lambda' : 1, #needs to be on order of magnitude as y
        "objective": "reg:absoluteerror",  
        'eval_metric' : 'mae',
        "device": "cuda"
    }
    # train model with both training and validation sets
    evals = [(dtrain, "train"), (dvalid, "valid")]
    evals_result = {}

    model = xgb.train(
        params=param,
        dtrain=dtrain,
        num_boost_round=500,
        evals=evals,
        early_stopping_rounds=50,
        evals_result=evals_result,
        verbose_eval=50
    )

    model.set_param({"device": "cuda"})
    dtotal = xgb.DMatrix(X, label=y, feature_names=list(X.columns))
    shap_values = model.predict(dtotal, pred_contribs=True)
    # shap_df = pd.DataFrame(shap_values, columns=list(X.columns) + ["bias"])


    explainer = shap.TreeExplainer(model)
    shap_values_full = explainer.shap_values(X)
    
    if len(y_cols) > 1:
        for j, output in enumerate(y.columns):
            for i, feature in enumerate(X.columns):
                plt.figure()
                shap.dependence_plot(
                    feature,
                    shap_values_full[:, :, j] / y.iloc[:, j].mean() * 100,
                    X,
                    show=False,
                    alpha=0.3
                )
                plt.ylabel(f'{output}')
                plt.show()
                
    else:
        for i,feature in enumerate(X.columns):
            plt.figure()
            shap.dependence_plot(feature, shap_values_full/y.mean() * 100, X, show = False, alpha = .3)
            plt.title(y_cols[0])
            plt.show()
        
    if save_path is not None:
        model.save_model(save_path)



def generate_classical_masks(files, 
                             species_regions, 
                             dayglow_model_pt = "cha_dayglow_xgb.json", 
                             save_path = r'D:\gold_l1c_NI\cha_2020_train',
                             channel = 'CHA'):
    
    '''Generates masks for gathering classical auroral masks. This loop will save
    the data: raw scans, differenced scans, and the auroral prediction masks.'''
    model = xgb.Booster()
    model.load_model(dayglow_model_pt)

    time_base = '2000-01-01-11-59'
    base_time = pd.to_datetime(time_base, format='%Y-%m-%d-%H-%M')

    out_dir = save_path
    os.makedirs(out_dir, exist_ok = True)
    
    data = []
    for f in tqdm(files):

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
            
            #DAYGLOW MODEL PREDICTS
            inp_data = np.stack([ema[valid_mask], sza[valid_mask]]).T
            inp_data = xgb.DMatrix(inp_data, feature_names=['ema', 'sza'])
            pred = model.predict(inp_data)
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
                
                raw = img[:,:,i]
                diff = np.clip(raw - (slopes[i]*pred_full[:,:,i] + biases[i]), 0, np.inf)
                differences.append(diff)
                dayglow.append((slopes[i]*pred_full[:,:,i] + biases[i]))
                if species[i] == '1493':
                    aurora = diff> 25
                else:
                    aurora = diff > 50
                
                cum_masks.append(aurora)
                
            #final mask operations by combining across species threshold.
            if  (int(f[-24:-21]) < 255) & (int(f[-24:-21]) > 115):
                final_mask = (np.stack(cum_masks).sum(axis = 0) > 2) * ~b1
            else:
                final_mask = (np.stack(cum_masks).sum(axis = 0) > 1) * ~b1
                
            dayglow = np.stack(dayglow).transpose(1,2,0)
            raw_scans = (img*(ema<90)[:,:,None])
            differences = np.stack(differences)
            
            
            # geo_info = np.dstack([lat.data, lon.data, sza.data, ema.data])
            # geo_info[np.isnan(geo_info)] = 0 
            # raw_scans = np.dstack([geo_info,raw_scans])

            raw_scans = np.dstack([dayglow,raw_scans])
            
            np.save(os.path.join(out_dir, f'raw_{f[-29:-15]}'), raw_scans.data)
            # np.save(os.path.join(out_dir, f'dayglow_{f[-29:-15]}'), dayglow)
            np.save(os.path.join(out_dir, f'mask_{f[-29:-15]}'), final_mask)
    
    
    