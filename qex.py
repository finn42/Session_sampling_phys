# functions for viewing, sorting, and triming the equivital sensor data files as exported by the RITMO installation of QIOSK

# put this in an early cell of any notebook useing these functions, uncommented. With starting %
# %load_ext autoreload
# %autoreload 1
# %aimport qex

import sys
import os
import time
import datetime as dt
import math
import numpy as np 
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# import heartpy as hp

from scipy.signal import butter,filtfilt
from scipy import interpolate
from scipy.interpolate import interp1d



def min_dets(eq_file_loc,sep): # for csv files output by the qiosk app
    if not sep:
        sep = '\\'
    filings = eq_file_loc.split(sep)
    file_name = filings[-1]
    f = file_name.split('-')
    Signal = f[0]
    DevName = f[1]#filings[-2]
    DevID = int(f[2])
    fileDate = int(f[3][:6]) # interpret as datetime datatype?
    # sometimes the session numbering fails and we get files with the same session number but an additiona _0 or _1
    # how to number this? What errors are producing these session numbers?
    if len(f[3].split('_'))==2: # we have an additional numbering to work into the sessions. :[
        Sessn1 = int(f[3].split('_')[0][6:8])
        Sessn2 = int(f[3].split('_')[1].split('.')[0])
        Session = (Sessn1+1)*100 + Sessn2+1 # yes this makes the session numbers huge out of nowhere, but it won't overlap with QIOSKs proper numbering that goes up to 99
    else:
        Session = int(f[3][6:8])
    fileSize = os.path.getsize(eq_file_loc)

    File_dets={'Signal':Signal, #f[-2].split('_')[-1],
       'DevName':DevName,
       'ID':DevID, 
       'Date':fileDate,
       'Session':Session,
       'FileName':file_name,
       'FileType':'csv',
       'FileSize': fileSize,
       'FullLoc':eq_file_loc}
    return File_dets

def data_dets(eq_file_loc,sep): #rec_start = V['DateTime'].iloc[0]
    if not sep:
        sep = '\\'
    # this file pulls recording details from the file name and from inside file to aggregate all metadata
    filings = eq_file_loc.split(sep)
    file_name = filings[-1]
    f = file_name.split('-')
    Signal = f[0]
    DevName = f[1]#filings[-2]
    DevID = int(f[2])
    fileDate = int(f[3][:6]) # interpret as datetime datatype?
    # sometimes the session numbering fails and we get files with the same session number but an additiona _0 or _1
    # how to number this? What errors are producing these session numbers?
    if len(f[3].split('_'))==2: # we have an additional numbering to work into the sessions. :[
        Sessn1 = int(f[3].split('_')[0][6:8])
        Sessn2 = int(f[3].split('_')[1].split('.')[0])
        Session = (Sessn1+1)*100 + Sessn2+1 # yes this makes the session numbers huge out of nowhere, but it won't overlap with QIOSKs proper numbering that goes up to 99
    else:
        Session = int(f[3][6:8])
    fileSize = os.path.getsize(eq_file_loc)
    
    eq_data_loc = sep.join(filings[:-1])
    M = matched_files(eq_file_loc,eq_data_loc,sep)
    mfiles = {'DATA':'','RESP':'','RR':'','ACC':'','ECG':''}
    for m in M:
        if 'DATA' in m:
            mfiles['DATA'] = m
        if 'RESPACC' in m:
            mfiles['RESP'] = m
        if 'RR' in m:
            mfiles['RR'] = m
        if 'FASTACC' in m:
            mfiles['ACC'] = m       
        if 'ECG' in m:
            mfiles['ECG'] = m 
            
    V = pd.read_csv(eq_file_loc,skipinitialspace=True)
    if len(V)==0:
        File_dets={'Signal':Signal, #f[-2].split('_')[-1],
           'DevName':DevName,
           'ID':DevID, 
           'Date':fileDate,
           'Session':Session,
           'FileName':file_name,
           'FileType':'csv',
           'FileSize': fileSize,
           'RecStart':pd.to_datetime('2020-02-02 02:02:00.00+0000'), # fake start
           'FullLoc':eq_file_loc,
            'DATAloc':mfiles['DATA'],
            'ACCloc': mfiles['ACC'],
            'RESPloc':mfiles['RESP'],
            'RRloc': mfiles['RR'],
            'ECGloc':mfiles['ECG']}
        return File_dets
    
    else:
        V['DateTime'] = pd.to_datetime(V['DateTime'])
        rec_start = V['DateTime'].iloc[0]
        rec_end = V['DateTime'].iloc[-1]
        rec_dur=(rec_end-rec_start).total_seconds()
        Batt_start = V['BATTERY(mV)'].iloc[0]
        Batt_end = V['BATTERY(mV)'].iloc[-1]
        Batt_spend=(Batt_end-Batt_start)     
        
        a = V.loc[:,['SENSOR ID', 'SUBJECT ID', 'SUBJECT AGE', 'HR(BPM)',
           'HRC(%)', 'BELT OFF', 'LEAD OFF', 'MOTION', 'BODY POSITION']].mode().loc[0]
        DevNames = V.loc[:,'SUBJECT ID'].unique()

        File_dets={'Signal':Signal, #f[-2].split('_')[-1],
           'DevName':DevName,
           'ID':DevID, 
           'Date':fileDate,
           'Session':Session,
           'FileName':file_name,
           'FileType':'csv',
           'FileSize': fileSize,
           'RecStart':rec_start,
           'RecEnd':rec_end,
           'Duration':rec_dur,
           'BatteryStart':Batt_start,
           'BatteryEnd':Batt_end,
           'BatteryChange(mV)':Batt_spend,
           'FullLoc':eq_file_loc,
            'DATAloc':mfiles['DATA'],
            'ACCloc': mfiles['ACC'],
            'RESPloc':mfiles['RESP'],
            'RRloc': mfiles['RR'],
            'ECGloc':mfiles['ECG'],
           'SubjectNames': DevNames}
        File_dets.update(a) # dic0.update(dic1)
        return File_dets

def test_plot_signals(V): # V is a qiosk file read into pandas
    if len(V)>2:
        V['DateTime'] = pd.to_datetime(V['DateTime'])
        W = V.select_dtypes(include=['int64','float64'])
        W.set_index(V['DateTime'],inplace=True)
        cols = W.columns
        # excerpt a minute of signal from the middle of the recording
        if V['DateTime'].iloc[-1]-V['DateTime'].iloc[0]>pd.to_timedelta(120,'s'):
            t1 =  V['DateTime'].iloc[int(len(V)/2)]
            t2 = t1+pd.to_timedelta(60,'s')
            X = W.loc[W.index>t1,:].copy()
            X = X.loc[X.index<t2,:].copy()
            for c in cols:
                fig, (ax1, ax2) = plt.subplots(1,2,figsize=[15,2])
                W[c].plot(ax=ax1)
                ax1.set_ylabel(c)
                X.loc[:,c].plot(ax=ax2)
                ax2.set_xlabel('60 seconds')
                plt.show()
        else:
            for c in cols:
                fig, (ax1) = plt.subplots(1,1,figsize=[15,2])
                W[c].plot(ax=ax1)
                ax1.set_ylabel(c)
                plt.show()
    else:
        print('Not enough data')
        
def test_plot_signal(V): # V is a qiosk file read into pandas
    if len(V)>2:
        if isinstance(V.index, pd.DatetimeIndex):
            V['DateTime'] = V.index
        V['DateTime'] = pd.to_datetime(V['DateTime'])
        W = V.select_dtypes(include=['int64','float64'])
        W.set_index(V['DateTime'],inplace=True)
        cols = W.columns
        print(cols[0])
        c = cols[0]
        # excerpt a minute of signal from the middle of the recording
        if V['DateTime'].iloc[-1]-V['DateTime'].iloc[0]>pd.to_timedelta(120,'s'):
            t1 =  V['DateTime'].iloc[int(len(V)/2)]
            t2 = t1+pd.to_timedelta(60,'s')
            X = W.loc[W.index>t1,:].copy()
            X = X.loc[X.index<t2,:].copy()
            #for c in cols[0]:
            fig, (ax1, ax2) = plt.subplots(1,2,figsize=[15,2])
            W[c].plot(ax=ax1)
            ax1.set_ylabel(c)
            X.loc[:,c].plot(ax=ax2)
            ax2.set_xlabel('60 seconds')
            plt.show()
        else:
           # for c in cols[0]:
            fig, (ax1) = plt.subplots(1,1,figsize=[15,2])
            W[c].plot(ax=ax1)
            ax1.set_ylabel(c)
            plt.show()
    else:
        print('Not enough data')
        
def cut_by_time(eq_file_loc,t1,t2):
    X = pd.DataFrame()
    if eq_file_loc.lower().endswith('csv'): 
        V = pd.read_csv(eq_file_loc,skipinitialspace=True)
        cols = V.columns
        if 'DateTime' in cols:
            V['DateTime'] = pd.to_datetime(V['DateTime'])
            V = remove_tz_from_dataframe(V)
            # if it made of dateTime stamps without millisecond values because the sampling is rare, force it!
#             if len(str(V.index[0]))<30: 
#                 V['DateTime'] = V['DateTime'] + dt.timedelta(milliseconds=5)
            V.set_index(V['DateTime'],inplace=True)
            V.drop(['DateTime'],axis=1, inplace=True)
            X = V.loc[V.index>t1,:].copy()
            X = X.loc[X.index<t2,:].copy()
            if len(X)<1:
                print('Recording does not intersect with that time interval')
                return
            else:
                return X
        else:
            print('Not a suitable format')
            return
    else:
        print('Not a CSV')
        return
        
def test_plot_signals_interval(V,t1,t2): # V is a qiosk file read into pandas
    # its on you to be sure these time stamps are within the recording interval of the file
    if len(V)>2:
        if isinstance(V.index, pd.DatetimeIndex):
            V['DateTime'] = V.index
        V['DateTime'] = pd.to_datetime(V['DateTime'])
        W = V.select_dtypes(include=['int64','float64'])
        W.set_index(V['DateTime'],inplace=True)
        cols = W.columns
        X = W.loc[W.index>t1,:]
        X = X.loc[X.index<t2,:].copy()
        for c in cols:
            fig, (ax1) = plt.subplots(1,1,figsize=[15,2])
            X.loc[:,c].plot(ax=ax1)
            ax1.set_ylabel(c)
            plt.show()
    else:
        print('No data') 

        
def test_plot_signals_interval_save(V,t1,t2,plotname): # V is a qiosk file read into pandas
    # its on you to be sure these time stamps are within the recording interval of the file
    if len(V)>2:
        if isinstance(V.index, pd.DatetimeIndex):
            V['DateTime'] = V.index
        V['DateTime'] = pd.to_datetime(V['DateTime'])
        W = V.select_dtypes(include=['int64','float64'])
        W.set_index(V['DateTime'],inplace=True)
        cols = W.columns
        X = W.loc[W.index>t1,:].copy()
        X = X.loc[X.index<t2,:].copy()
        for c in cols:
            fig, (ax1) = plt.subplots(1,1,figsize=[15,2])
            X.loc[:,c].plot(ax=ax1)
            ax1.set_ylabel(c)
            plt.savefig("./plots/"+plotname+"_"+c+"_seg.png",bbox_inches = 'tight',dpi = 300)
            plt.show()
    else:
        print('No data') 
        
def matched_files(eq_file_loc,data_path,sep):
    if not sep:
        sep = '\\'
    # from the location of a good file and the location of other files, retrieve the location of all matching files
    dfile = min_dets(eq_file_loc,sep)
    
    # retrieve the files in that path that match 
    file_locs = []
    for root, dirs, files in os.walk(data_path):
        for file in files:
            if(file.lower().endswith(".csv")):
                file_locs.append(os.path.join(root,file))
    k=[]
    for file in file_locs:
        if not file.lower().endswith('recordings.csv'):
#             print(file)
            File_dets=min_dets_sem(file,sep)
            k.append(File_dets)
    df_files=pd.DataFrame(data=k)
#     print(df_files)

    match_fields = ['ID','DevName','Date','Session']

    matched_files = df_files.loc[df_files['ID'] == dfile['ID']]
    for mf in match_fields[1:]:
        matched_files = matched_files.loc[matched_files[mf] == dfile[mf]]

    return list(matched_files['FullLoc'])+list(matched_files['SEMLoc'].unique())
    

def min_dets_sem(eq_file_loc,sep): # for files output by the lab manager desktop app, so far
    w = eq_file_loc.split(sep)
    file_name = w[-1] #ECG-Pilot_1-3420755-23020100-trimmed.csv'
    f = file_name.split('-')
    Signal = f[0]
    DevName = f[1]#filings[-2]
    DevID = int(f[2])
    fileDate = int(f[3][:6]) # interpret as datetime datatype?
    # sometimes the session numbering fails and we get files with the same session number but an additiona _0 or _1
    # how to number this? What errors are producing these session numbers?
    if len(f[3].split('_'))==2: # we have an additional numbering to work into the sessions. :[
        Sessn1 = int(f[3].split('_')[0][6:8])
        Sessn2 = int(f[3].split('_')[1].split('.')[0])
        Session = (Sessn1+1)*100 + Sessn2+1 # yes this makes the session numbers huge out of nowhere, but it won't overlap with QIOSKs proper numbering that goes up to 99
    else:
        Session = int(f[3][6:8])
    fileSize = os.path.getsize(eq_file_loc)
    
    if len(f)<5: # no trimming feilds included, so sem file
        # assuming qiosks file structure is consistent, SEM file naming should be reliable
        if eq_file_loc.startswith('C:\\Users\\Public\\Documents\\Equivital\\Equivital Manager Wizard\\'): # initial exports
            fn = w[-1].split('-')[-1][:-3]+'SEM'
            sem_path = w[:-3]+['Raw SEM Data',w[-2],fn]
            sem_loc = sep.join(sem_path)
        else:
            if w[-2].lower().startswith('csv'): # Use earlier details to formulate location and structure
                fn = file_name.split('-')[-1][:-3]+'SEM'
                sem_path = eq_file_loc.split(sep)[:-2] + ['SEM',DevName,fn]
                sem_loc = sep.join(sem_path)
            else:
                sem_loc = ''
    else:
        sem_loc = ''
    File_dets={'Signal':Signal, #f[-2].split('_')[-1],
       'DevName':DevName,
       'ID':DevID, 
       'Date':fileDate,
       'Session':Session,
       'FileName':file_name,
       'FileType':'csv',
       'FileSize': fileSize,
       'FullLoc':eq_file_loc,
       'SEMLoc': sem_loc}
    return File_dets

def qiosk_recordings(projectpath,projecttag,sep):
    file_locs = []
    for root, dirs, files in os.walk(projectpath):
        for file in files:
            if(file.lower().endswith(".csv")):
                if file.lower().startswith('data'):
                    file_locs.append(os.path.join(root,file))
    if len(file_locs)>0:
        k=[]           
        for f in file_locs:
            File_dets=data_dets(f,sep)
            if File_dets:
                k.append(File_dets)
        df_datafiles=pd.DataFrame(data=k)#
        df_datafiles=df_datafiles.sort_values(by='RecStart').reset_index(drop=True)
        df_datafiles.to_csv(projectpath + projecttag + '_Qiosk_recordings.csv')
        return df_datafiles
    else:
        print('Path is empty of DATA files.')
        return []
    
def qiosk_rec_check(projectpath,projecttag,sep):
    file_locs = []
    for root, dirs, files in os.walk(projectpath):
        for file in files:
            if(file.lower().endswith(".csv")):
                if file.lower().startswith('data'):
                    file_locs.append(os.path.join(root,file))
    if len(file_locs)>0:
        k=[]           
        for f in file_locs:
            File_dets=data_dets(f,sep)
            if File_dets:
                k.append(File_dets)
        df_datafiles=pd.DataFrame(data=k)#
        df_datafiles=df_datafiles.sort_values(by='RecStart').reset_index(drop=True)
        return df_datafiles
    else:
        print('Path is empty of DATA files.')
        return []
    
def remove_tz_from_dataframe(df_in):
    df = df_in.copy()
    col_times = [ col for col in df.columns if any([isinstance(x, pd.Timestamp) for x in df[col]])]
    for col in col_times:
        df[col] = pd.to_datetime(df[col],format='ISO8601') 
        df[col] = df[col].dt.tz_localize(None) 
    return df
