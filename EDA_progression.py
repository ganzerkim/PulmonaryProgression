# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 17:16:15 2020

@author: MG
"""
#%%
import warnings
warnings.filterwarnings('ignore')

import os

import numpy as np
import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

from scipy import stats

import matplotlib.pyplot as plt
import seaborn as sns

#%%
df_train = pd.read_csv('C:/Users/MG/Desktop/OSIC_pre/train.csv')
df_test = pd.read_csv('C:/Users/MG/Desktop/OSIC_pre/test.csv')
print(f'Training Set Shape = {df_train.shape} - Patients = {df_train["Patient"].nunique()}')
print(f'Training Set Memory Usage = {df_train.memory_usage().sum() / 1024 ** 2:.2f} MB')
print(f'Test Set Shape = {df_test.shape} - Patients = {df_test["Patient"].nunique()}')
print(f'Test Set Memory Usage = {df_test.memory_usage().sum() / 1024 ** 2:.2f} MB')

#%%
training_sample_counts = df_train.rename(columns={'Weeks': 'Samples'}).groupby('Patient').agg('count')['Samples'].value_counts()
print(f'Training Set FVC Measurements Per Patient \n{("-") * 41}\n{training_sample_counts}')

#%%
print(f'FVC Statistical Summary\n{"-" * 23}')

print(f'Mean: {df_train["FVC"].mean():.6}  -  Median: {df_train["FVC"].median():.6}  -  Std: {df_train["FVC"].std():.6}')
print(f'Min: {df_train["FVC"].min()}  -  25%: {df_train["FVC"].quantile(0.25)}  -  50%: {df_train["FVC"].quantile(0.5)}  -  75%: {df_train["FVC"].quantile(0.75)}  -  Max: {df_train["FVC"].max()}')
print(f'Skew: {df_train["FVC"].skew():.6}  -  Kurtosis: {df_train["FVC"].kurtosis():.6}')
missing_values_count = df_train[df_train["FVC"].isnull()].shape[0]
training_samples_count = df_train.shape[0]
print(f'Missing Values: {missing_values_count}/{training_samples_count} ({missing_values_count * 100 / training_samples_count:.4}%)')

fig, axes = plt.subplots(ncols=2, figsize=(18, 6), dpi=150)

sns.distplot(df_train['FVC'], label='FVC', ax=axes[0])
stats.probplot(df_train['FVC'], plot=axes[1])

for i in range(2):
    axes[i].tick_params(axis='x', labelsize=12)
    axes[i].tick_params(axis='y', labelsize=12)
    axes[i].set_xlabel('')
    axes[i].set_ylabel('')
    
axes[0].set_title(f'FVC Distribution in Training Set', size=15, pad=15)
axes[1].set_title(f'FVC Probability Plot', size=15, pad=15)

plt.show()

#%%
def plot_fvc(df, patient):
        
    df[['Weeks', 'FVC']].set_index('Weeks').plot(figsize=(30, 6), label='_nolegend_')
    
    plt.tick_params(axis='x', labelsize=20)
    plt.tick_params(axis='y', labelsize=20)
    plt.xlabel('')
    plt.ylabel('')
    plt.title(f'Patient: {patient} - {df["Age"].tolist()[0]} - {df["Sex"].tolist()[0]} - {df["SmokingStatus"].tolist()[0]} ({len(df)} Measurements in {(df["Weeks"].max() - df["Weeks"].min())} Weeks Period)', size=25, pad=25)
    plt.legend().set_visible(False)
    plt.show()


for patient, df in list(df_train[0:16].groupby('Patient')):
    
    df['FVC_diff-1'] = np.abs(df['FVC'].diff(-1))
    
    print(f'Patient: {patient} FVC Statistical Summary\n{"-" * 58}')
    print(f'Mean: {df["FVC"].mean():.6}  -  Median: {df["FVC"].median():.6}  -  Std: {df["FVC"].std():.6}')
    print(f'Min: {df["FVC"].min()} -  Max: {df["FVC"].max()}')
    print(f'Skew: {df["FVC"].skew():.6}  -  Kurtosis: {df["FVC"].kurtosis():.6}')
    print(f'Change Mean: {df["FVC_diff-1"].mean():.6}  - Change Median: {df["FVC_diff-1"].median():.6}  - Change Std: {df["FVC_diff-1"].std():.6}')
    print(f'Change Min: {df["FVC_diff-1"].min()} -  Change Max: {df["FVC_diff-1"].max()}')
    print(f'Change Skew: {df["FVC_diff-1"].skew():.6} -  Change Kurtosis: {df["FVC_diff-1"].kurtosis():.6}')
    
    plot_fvc(df, patient)

















