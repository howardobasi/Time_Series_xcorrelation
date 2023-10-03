#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  1 01:37:42 2023

@author: howardobasi
"""

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt 
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from scipy import signal
 


'''importing data'''

BTC_1D = pd.read_csv('BTC-1DS.csv') # Read in the daily BTC data and extract the closing price
BTC_1D_CLOSE = BTC_1D['4']
D_TIME = pd.to_datetime(BTC_1D['Timestamp']) # Convert the daily timestamp data to datetime format
ETH_1D = pd.read_csv('ETH-1DS.csv')          # ETH ^
ETH_1D_CLOSE = ETH_1D['4']
ADA_1D = pd.read_csv('ADA-1DS.csv')          # ADA
ADA_1D_CLOSE = ADA_1D['4']


BTC_1H = pd.read_csv('BTC-1HS.csv',header = 0,nrows=730) # same as above but for hourly data
BTC_1H_CLOSE = BTC_1H['4']
H_TIME = pd.to_datetime(BTC_1H['Timestamp'])
ETH_1H = pd.read_csv('ETH-1HS.csv',header = 0,nrows=730)
ETH_1H_CLOSE = ETH_1H['4']
ADA_1H = pd.read_csv('ADA-1HS.csv',header = 0,nrows=730)
ADA_1H_CLOSE = ADA_1H['4']


BTC_1M = pd.read_csv('BTC-1MS.csv',header = 0,nrows=730) # same as above but for minute data
BTC_1M_CLOSE = BTC_1M['4']
M_TIME = pd.to_datetime(BTC_1M['Timestamp'])
ETH_1M = pd.read_csv('ETH-1MS.csv',header = 0,nrows=730)
ETH_1M_CLOSE = ETH_1M['4']
ADA_1M = pd.read_csv('ADA-1MS.csv',header = 0,nrows=730)
ADA_1M_CLOSE = ADA_1M['4']


''' functions '''

def normalise_price(close_price):
    
    norm_price = close_price / close_price.max() # Normalize the close prices by dividing each value by the maximum value in the series
    
    return norm_price # Return the normalized price series

def inital_plot(x,y1,y2,y3,label1,label2,label3,form,step):
    
    plt.plot(x,y1, label = label1,linewidth = 1) # plot series 1
    plt.plot(x,y2, label = label2,linewidth = 1) # plot series 2
    plt.plot(x,y3, label = label3,linewidth = 1) # plot series 3
    plt.grid(axis='both') 
    plt.xlabel('Time')
    plt.ylabel('normalised price') 
    # customize the x-axis tick labels
    x_tick_labels = x[::step]  # get every 60th tick label 
    x_tick_labels = [x.strftime(form) for x in x_tick_labels]  # format as YYYY-MM-DD
    plt.xticks(x[::step], x_tick_labels, rotation=45, ha='right')  # set tick locations and labels
    plt.legend() # plot the legend
    plt.show()

def corr_matrix(x1,y1,z1,x2,y2,z2):
    
    corr_matrix1 = pd.DataFrame({'BTC': [1.0, x1, y1], # Construct Pearson correlation matrix for dataset 1
                                'ETH': [x1, 1.0, z1],
                                'ADA': [y1, z1, 1.0]},
                             index=['BTC', 'ETH', 'ADA'])
    
    corr_matrix2 = pd.DataFrame({'BTC': [1.0, x2, y2],  # Construct Spearman rank correlation matrix for dataset 2
                                'ETH': [x2, 1.0, z2],
                                'ADA': [y2, z2, 1.0]},
                             index=['BTC', 'ETH', 'ADA'])
    
    fig, axs = plt.subplots(1, 2, figsize=(10, 5)) # Create a figure with two subplots
    
    sns.heatmap(corr_matrix1, annot=True, cmap='coolwarm', square=True, vmin=-1, vmax=1, ax=axs[0])
    axs[0].set_title("Pearson Correlation Matrix")    # Create a heatmap of the Pearson correlation matrix in the first subplot
    
    sns.heatmap(corr_matrix2, annot=True, cmap='coolwarm', square=True, vmin=-1, vmax=1, ax=axs[1])
    axs[1].set_title("Spearmen rank Correlation Matrix")    # Create a heatmap of the Spearmen correlation matrix in the second subplot
    
    plt.show()

def xcorr(x,y):
    """
Computes the cross-correlation of two input signals x and y.

Parameters:
    x (array-like): The first input signal.
    y (array-like): The second input signal.

Returns:
    correlation (array-like): The cross-correlation of the two input signals.
    lags (array-like): The lags associated with the cross-correlation values.
    lag (float): The time delay that maximizes the cross-correlation.

"""
    x2 = np.array(x)  # Convert input signals to NumPy arrays

    y2 = np.array(y)


    correlation = signal.correlate(x2-np.mean(x2), y2 - np.mean(y2), mode="full",method='direct')  # Compute the cross-correlation using scipy.signal.correlate function
    lags = signal.correlation_lags(len(x2), len(y2), mode="full")  # Compute the lags associated with the cross-correlation values
    lag = lags[np.argmax(abs(correlation))]   # Compute the time delay that maximizes the cross-correlation
    
    return correlation,lags,lag  # Return the cross-correlation, lags, and time delay

def smooth_ema(x,span):
    
    ema = x.ewm(span).mean() # Exponential moving average using pandas' ewm function
    
    return ema # Return the smoothed data

def lagg(x,y,z,r11,r12,r21,r22,r31,r32 ):
    """
  Compute cross-correlations for three time series and return the time lags 
  where the correlations are maximal
  
  Parameters:
  x: a list of 3 pandas dataframes, each containing time series data for a different cryptocurrency
  y: a list of 3 pandas dataframes, each containing time series data for a different cryptocurrency
  z: a list of 3 pandas dataframes, each containing time series data for a different cryptocurrency
  r11, r12, r21, r22, r31, r32: integer values representing the range of the time series data to analyze
  
  Returns:
  lagg_df: a pandas dataframe containing the time lags where cross-correlations are maximal
  
  """  
    # Compute cross-correlations between pairs of time series for each cryptocurrency
    c,ls,l1=xcorr(x[0].iloc[r11:r12],y[0].iloc[r11:r12])
    c,ls,l2=xcorr(x[0].iloc[r11:r12],z[0].iloc[r11:r12])
    c,ls,l3=xcorr(y[0].iloc[r11:r12],z[0].iloc[r11:r12])
    
    c,ls,l4=xcorr(x[1].iloc[r21:r22],y[1].iloc[r21:r22])
    c,ls,l5=xcorr(x[1].iloc[r21:r22],z[1].iloc[r21:r22])
    c,ls,l6=xcorr(y[1].iloc[r21:r22],z[1].iloc[r21:r22])
    
    c,ls,l7=xcorr(x[2].iloc[r31:r32],y[2].iloc[r31:r32])
    c,ls,l8=xcorr(x[2].iloc[r31:r32],z[2].iloc[r31:r32])
    c,ls,l9=xcorr(y[2].iloc[r31:r32],z[2].iloc[r31:r32])

    
     # Create a dataframe to store the time lags where cross-correlations are maximal
    lagg_df = pd.DataFrame({'BTC-ETH': [l1, l4, l7],
                            'BTC-ADA': [l2, l5, l8],
                            'ETH-ADA': [l3, l6, l9]},
                                index=['1', '2', '3'])       
    return lagg_df
         
''' normalising price '''

BTC_1D_NORM = normalise_price(BTC_1D_CLOSE)
ETH_1D_NORM = normalise_price(ETH_1D_CLOSE)
ADA_1D_NORM = normalise_price(ADA_1D_CLOSE)

BTC_1H_NORM = normalise_price(BTC_1H_CLOSE)
ETH_1H_NORM = normalise_price(ETH_1H_CLOSE)
ADA_1H_NORM = normalise_price(ADA_1H_CLOSE)

BTC_1M_NORM = normalise_price(BTC_1M_CLOSE)
ETH_1M_NORM = normalise_price(ETH_1M_CLOSE)
ADA_1M_NORM = normalise_price(ADA_1M_CLOSE)


''' initial plotting of data '''

inital_plot(D_TIME,BTC_1D_NORM,ETH_1D_NORM,ADA_1D_NORM,'BTC-1D','ETH-1D','ADA-1D','%Y-%m-%d',66)

inital_plot(H_TIME,BTC_1H_NORM,ETH_1H_NORM,ADA_1H_NORM,'BTC-1H','ETH-1H','ADA-1H','%Y-%m-%d',66)

inital_plot(M_TIME,BTC_1M_NORM,ETH_1M_NORM,ADA_1M_NORM,'BTC-1M','ETH-1M','ADA-1M','%H:%M',60)


''' Pearson's corelation '''

corr1, p_val1 = pearsonr(BTC_1D_NORM, ETH_1D_NORM)
corr2, p_val2 = pearsonr(BTC_1D_NORM, ADA_1D_NORM)
corr3, p_val3 = pearsonr(ETH_1D_NORM, ADA_1D_NORM)
corr4, p_val4 = pearsonr(BTC_1H_NORM, ETH_1H_NORM)
corr5, p_val5 = pearsonr(BTC_1H_NORM, ADA_1H_NORM)
corr6, p_val6 = pearsonr(ETH_1H_NORM, ADA_1H_NORM)
corr7, p_val7 = pearsonr(BTC_1M_NORM, ETH_1M_NORM)
corr8, p_val8 = pearsonr(BTC_1M_NORM, ADA_1M_NORM)
corr9, p_val9 = pearsonr(ETH_1M_NORM, ADA_1M_NORM)

pearson_df = pd.DataFrame({'BTC-ETH': [corr1, corr4, corr7],
                           'BTC-ADA': [corr2, corr5, corr8],
                           'ETH-ADA': [corr3, corr6, corr9]},
                          index=['1D', '1H', '1M'])

pearson_pval_df = pd.DataFrame({'BTC-ETH': [p_val1, p_val4, p_val7],
                                'BTC-ADA': [p_val2, p_val5, p_val8],
                                'ETH-ADA': [p_val3, p_val6, p_val9]},
                               index=['1D', '1H', '1M'])

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
print()
print('Pearson & p values')
print()
print(pearson_df)
print()
print(pearson_pval_df)
print()


''' spearmen's rank correlation '''                    
# Calculate the Pearson's correlation coefficients and p-values for all pairs of cryptocurrencies and timeframes.
scorr1, sp_val1 = spearmanr(BTC_1D_NORM, ETH_1D_NORM)
scorr2, sp_val2 = spearmanr(BTC_1D_NORM, ADA_1D_NORM)
scorr3, sp_val3 = spearmanr(ETH_1D_NORM, ADA_1D_NORM)
scorr4, sp_val4 = spearmanr(BTC_1H_NORM, ETH_1H_NORM)
scorr5, sp_val5 = spearmanr(BTC_1H_NORM, ADA_1H_NORM)
scorr6, sp_val6 = spearmanr(ETH_1H_NORM, ADA_1H_NORM)
scorr7, sp_val7 = spearmanr(BTC_1M_NORM, ETH_1M_NORM)
scorr8, sp_val8 = spearmanr(BTC_1M_NORM, ADA_1M_NORM)
scorr9, sp_val9 = spearmanr(ETH_1M_NORM, ADA_1M_NORM)

spearman_df = pd.DataFrame({'BTC-ETH': [scorr1, scorr4, scorr7], # Create a DataFrame to store the Pearson's correlation coefficients for all pairs of cryptocurrencies and timeframes.
                            'BTC-ADA': [scorr2, scorr5, scorr8],
                            'ETH-ADA': [scorr3, scorr6, scorr9]},
                           index=['1D', '1H', '1M'])

spearman_pval_df = pd.DataFrame({'BTC-ETH': [sp_val1, sp_val4, sp_val7], # Create a DataFrame to store the p-values for all pairs of cryptocurrencies and timeframes.
                                 'BTC-ADA': [sp_val2, sp_val5, sp_val8],
                                 'ETH-ADA': [sp_val3, sp_val6, sp_val9]},
                                index=['1D', '1H', '1M'])
print('Spearmen & p values')
print()
print(spearman_df)
print()
print(spearman_pval_df)
print()


''' correlation matrix '''

corr_matrix(corr1,corr2,corr3,scorr1,scorr2,scorr3)

corr_matrix(corr4,corr5,corr6,scorr4,scorr5,scorr6)

corr_matrix(corr7,corr8,corr9,scorr7,scorr8,scorr9)


''' cross correlation (lag test 1) '''

#correlation,lags,lag=xcorr(ETH_1M_NORM,ADA_1M_NORM)   # cross correlation for entire data (each outlook)
#print(lag)


''' smoothing the data '''

BTC_1D_SMOOTH = smooth_ema(BTC_1D_NORM,5)
BTC_1H_SMOOTH = smooth_ema(BTC_1H_NORM,5)
BTC_1M_SMOOTH = smooth_ema(BTC_1M_NORM,5)

ETH_1D_SMOOTH = smooth_ema(ETH_1D_NORM,5)
ETH_1H_SMOOTH = smooth_ema(ETH_1H_NORM,5)
ETH_1M_SMOOTH = smooth_ema(ETH_1M_NORM,5)

ADA_1D_SMOOTH = smooth_ema(ADA_1D_NORM,5)
ADA_1H_SMOOTH = smooth_ema(ADA_1H_NORM,5)
ADA_1M_SMOOTH = smooth_ema(ADA_1M_NORM,5)

inital_plot(D_TIME,BTC_1D_SMOOTH,ETH_1D_SMOOTH,ADA_1D_SMOOTH,'BTC-1D','ETH-1D','ADA-1D','%Y-%m-%d',66)

inital_plot(H_TIME,BTC_1H_SMOOTH,ETH_1H_SMOOTH,ADA_1H_SMOOTH,'BTC-1H','ETH-1H','ADA-1H','%Y-%m-%d',66)

inital_plot(M_TIME,BTC_1M_SMOOTH,ETH_1M_SMOOTH,ADA_1M_SMOOTH,'BTC-1M','ETH-1M','ADA-1M','%H:%M',60)


''' seperating each outlook '''

D_TIME_S = np.array_split(D_TIME, 3)   # splitting thw time series into 3 equal sections
H_TIME_S = np.array_split(H_TIME, 3)
M_TIME_S = np.array_split(M_TIME, 3)

BTC_1D_SS = np.array_split(BTC_1D_SMOOTH, 3)
BTC_1H_SS = np.array_split(BTC_1H_SMOOTH, 3)
BTC_1M_SS = np.array_split(BTC_1M_SMOOTH, 3)


ETH_1D_SS = np.array_split(ETH_1D_SMOOTH, 3)
ETH_1H_SS = np.array_split(ETH_1H_SMOOTH, 3)
ETH_1M_SS = np.array_split(ETH_1M_SMOOTH, 3)


ADA_1D_SS = np.array_split(ADA_1D_SMOOTH, 3)
ADA_1H_SS = np.array_split(ADA_1H_SMOOTH, 3)
ADA_1M_SS = np.array_split(ADA_1M_SMOOTH, 3)



''' cross correlation (lag test 2)'''

print('1-D LAG')
print(lagg(BTC_1D_SS,ETH_1D_SS,ADA_1D_SS,110,185,90,150,70,120)) # using the lagg function computs lag for relevant pairs for chosen periods 
print()
print('1-H LAG')
print(lagg(BTC_1H_SS,ETH_1H_SS,ADA_1H_SS,144,200,150,210,25,70))
print()
print('1-M LAG')
print(lagg(BTC_1M_SS,ETH_1M_SS,ADA_1M_SS,195,280,160,220,160,240))
print()



''' error analysis '''


c,ls,l = xcorr(BTC_1M_SS[2],ADA_1M_SS[2])  # compute cross correlation 

plt.plot(ls,c)                            # plot cross correlation distribution 
plt.grid(axis='both')
plt.xticks(np.arange(-100,100,20))
plt.xlim(-100,100)
plt.ylim(0,0.005)
plt.ylabel('Cross - correlation')
plt.xlabel('lag (minutes)')
plt.vlines(l,ymax=0.0042,ymin=0,color='r')
plt.show

n = len(c)
r = np.max(abs(c))
variance = (1 - r**2) / (n - 2)        # standard error calcualtion on cross calcualtion
std_error = np.sqrt(variance)
lag_error = (std_error) * (1 * n)**(-1/2)    # propogation of errors for corresponding error on lag
print("Standard error on lag: ", lag_error) 




''' final plotting '''  # final olots are below ignore if needed 


gra = np.arange(0,3,1)


fig, axs = plt.subplots(2, 3, figsize=(10, 5), sharey=True)
fig.subplots_adjust(wspace=0.03,hspace=0.03)

for i in gra:    
 axs[0,i].plot(D_TIME_S[i],BTC_1D_SS[i],label='BTC',color='#1f77b4')
 axs[0,i].plot(D_TIME_S[i],ETH_1D_SS[i],label='ETH',color='#ff7f0e')
 axs[0,i].plot(D_TIME_S[i],ADA_1D_SS[i],label='ADA',color='#2ca02c')
 axs[0,i].grid(True)
 axs[0, i].set_xticklabels([])
 axs[0,2].legend()
 axs[0,0].set_ylabel('normalised price')
 
 axs[1,0].plot(D_TIME_S[0].iloc[110:185],BTC_1D_SS[0].iloc[110:185],color='#1f77b4')
 axs[1,0].plot(D_TIME_S[0].iloc[110:185],ETH_1D_SS[0].iloc[110:185],color='#ff7f0e')
 axs[1,0].plot(D_TIME_S[0].iloc[110:185],ADA_1D_SS[0].iloc[110:185],color='#2ca02c')
 
 axs[1,1].plot(D_TIME_S[1].iloc[90:150],BTC_1D_SS[1].iloc[90:150],color='#1f77b4')
 axs[1,1].plot(D_TIME_S[1].iloc[90:150],ETH_1D_SS[1].iloc[90:150].shift(-2),color='#ff7f0e')
 axs[1,1].plot(D_TIME_S[1].iloc[90:150],ADA_1D_SS[1].iloc[90:150].shift(-1),color='#2ca02c')
 
 axs[1,2].plot(D_TIME_S[2].iloc[70:120],BTC_1D_SS[2].iloc[70:120],color='#1f77b4')
 axs[1,2].plot(D_TIME_S[2].iloc[70:120],ETH_1D_SS[2].iloc[70:120],color='#ff7f0e')
 axs[1,2].plot(D_TIME_S[2].iloc[70:120],ADA_1D_SS[2].iloc[70:120],color='#2ca02c')
 
 axs[1,0].set_xlim(axs[0,0].get_xlim())
 axs[1,1].set_xlim(axs[0,1].get_xlim())
 axs[1,2].set_xlim(axs[0,2].get_xlim())
 
 axs[1, i].tick_params(axis='x', labelrotation=45)
 axs[1,i].grid(True)
 axs[1,0].set_ylabel('normalised price') 
 axs[1,1].set_xlabel('Time (YYYY-MM)')
 
plt.show()



fig, axs = plt.subplots(2, 3, figsize=(10, 5), sharey=True)
fig.subplots_adjust(wspace=0.03,hspace=0.03)

for i in gra:    
 axs[0,i].plot(H_TIME_S[i],BTC_1H_SS[i],label='BTC',color='#1f77b4')
 axs[0,i].plot(H_TIME_S[i],ETH_1H_SS[i],label='ETH',color='#ff7f0e')
 axs[0,i].plot(H_TIME_S[i],ADA_1H_SS[i],label='ADA',color='#2ca02c')
 axs[0,i].grid(True)
 axs[0, i].set_xticklabels([])
 axs[0,2].legend()
 axs[0,0].set_ylabel('normalised price')
 
 axs[1,0].plot(H_TIME_S[0].iloc[144:200],BTC_1H_SS[0].iloc[144:200],color='#1f77b4')
 axs[1,0].plot(H_TIME_S[0].iloc[144:200],ETH_1H_SS[0].iloc[144:200],color='#ff7f0e')
 axs[1,0].plot(H_TIME_S[0].iloc[144:200],ADA_1H_SS[0].iloc[144:200],color='#2ca02c')
 
 axs[1,1].plot(H_TIME_S[1].iloc[150:210],BTC_1H_SS[1].iloc[150:210],color='#1f77b4')
 axs[1,1].plot(H_TIME_S[1].iloc[150:210],ETH_1H_SS[1].iloc[150:210],color='#ff7f0e')
 axs[1,1].plot(H_TIME_S[1].iloc[150:210],ADA_1H_SS[1].iloc[150:210],color='#2ca02c')
 
 axs[1,2].plot(H_TIME_S[2].iloc[25:70],BTC_1H_SS[2].iloc[25:70],color='#1f77b4')
 axs[1,2].plot(H_TIME_S[2].iloc[25:70],ETH_1H_SS[2].iloc[25:70].shift(-1),color='#ff7f0e')
 axs[1,2].plot(H_TIME_S[2].iloc[25:70],ADA_1H_SS[2].iloc[25:70],color='#2ca02c')
 
 axs[1,0].set_xlim(axs[0,0].get_xlim())
 axs[1,1].set_xlim(axs[0,1].get_xlim())
 axs[1,2].set_xlim(axs[0,2].get_xlim())
 
 axs[1, i].tick_params(axis='x', labelrotation=45)
 axs[1,i].grid(True)
 axs[1,0].set_ylabel('normalised price') 
 axs[1,1].set_xlabel('Time (YYYY-MM-DD)')
 
 
plt.show()



fig, axs = plt.subplots(2, 3, figsize=(10, 5), sharey=True)
fig.subplots_adjust(wspace=0.03,hspace=0.03)

for i in gra:    
 axs[0,i].plot(M_TIME_S[i],BTC_1M_SS[i],label='BTC',color='#1f77b4')
 axs[0,i].plot(M_TIME_S[i],ETH_1M_SS[i],label='ETH',color='#ff7f0e')
 axs[0,i].plot(M_TIME_S[i],ADA_1M_SS[i],label='ADA',color='#2ca02c')
 axs[0,i].grid(True)
 axs[0, i].set_xticklabels([])
 axs[0,2].legend()
 axs[0,0].set_ylabel('normalised price')
 
 axs[1,0].plot(M_TIME_S[0].iloc[195:],BTC_1M_SS[0].iloc[195:].shift(-1),color='#1f77b4')
 axs[1,0].plot(M_TIME_S[0].iloc[195:],ETH_1M_SS[0].iloc[195:].shift(),color='#ff7f0e')
 axs[1,0].plot(M_TIME_S[0].iloc[195:],ADA_1M_SS[0].iloc[195:].shift(),color='#2ca02c')
 
 axs[1,1].plot(M_TIME_S[1].iloc[160:220],BTC_1M_SS[1].iloc[160:220],color='#1f77b4')
 axs[1,1].plot(M_TIME_S[1].iloc[160:220],ETH_1M_SS[1].iloc[160:220],color='#ff7f0e')
 axs[1,1].plot(M_TIME_S[1].iloc[160:220],ADA_1M_SS[1].iloc[160:220],color='#2ca02c')

 axs[1,2].plot(M_TIME_S[2].iloc[160:240],BTC_1M_SS[2].iloc[160:240],color='#1f77b4')
 axs[1,2].plot(M_TIME_S[2].iloc[160:240],ETH_1M_SS[2].iloc[160:240],color='#ff7f0e')
 axs[1,2].plot(M_TIME_S[2].iloc[160:240],ADA_1M_SS[2].iloc[160:240].shift(-1),color='#2ca02c')
 
 axs[1,0].set_xlim(axs[0,0].get_xlim())
 axs[1,1].set_xlim(axs[0,1].get_xlim())
 axs[1,2].set_xlim(axs[0,2].get_xlim())
 
 axs[1, i].tick_params(axis='x', labelrotation=45)
 axs[1,i].grid(True)
 axs[1,0].set_ylabel('normalised price') 
 axs[1,1].set_xlabel('Time (DD-HH-MIN)')
 
 
plt.show()










