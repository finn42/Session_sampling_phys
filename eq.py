# functions for viewing, sorting, and triming the equivital sensor data files for the Bodies on Concert project, after file names have been fix.


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


def eq_recordings(projectpath,projecttag,sep):
    file_locs = []
    for root, dirs, files in os.walk(projectpath):
        for file in files:
            if(file.endswith("EQDATA.csv")):
                if(file.lower().startswith(projecttag.lower())):
                    file_locs.append(os.path.join(root,file))
    if len(file_locs)>0:
        k=[]           
        for f in file_locs:
            File_dets=data_dets(f,sep)
            if File_dets:
                k.append(File_dets)
        df_datafiles=pd.DataFrame(data=k)#
        df_datafiles=df_datafiles.sort_values(by='RecStart').reset_index(drop=True)
        #df_datafiles.to_csv(projectpath + projecttag + '_Qiosk_recordings.csv')
        return df_datafiles
    else:
        print('Path is empty of DATA files.')
        return []
# fN ='_'.join([file_date,Concert,Piece,PartID,sigtype])+file_type
# /Users/finn/Desktop/Current_Projects/Stavanger/Data/Equivital/Reic/C2/2023-02-16_C1_REIC_VN205_EQDATA.csv
def data_dets(eq_file_loc,sep): #rec_start = V['DateTime'].iloc[0]
    if not sep:
        sep = '\\'
    # this file pulls recording details from the file name and from inside file to aggregate all metadata
    filings = eq_file_loc.split(sep)
    file_name = filings[-1]
    f = file_name.split('_')
    Signal = f[4][2:-4]
    DevName = f[3]#filings[-2]
    Concert = f[1]#filings[-2]
    Work = f[2]#filings[-2]
    fileDate = f[0] # interpret as datetime datatype?
    fileSize = os.path.getsize(eq_file_loc)
    
    V = pd.read_csv(eq_file_loc,skipinitialspace=True)
    
    V['DateTime'] = pd.to_datetime(V['DateTime'])
    rec_start = V['DateTime'].iloc[0]
    rec_end = V['DateTime'].iloc[-1]
    rec_dur=(rec_end-rec_start).total_seconds()
    Batt_start = V['BATTERY(mV)'].iloc[0]
    Batt_end = V['BATTERY(mV)'].iloc[-1]
    Batt_spend=(Batt_end-Batt_start)     

    a = V.loc[:,['SENSOR ID', 'SUBJECT ID', 'SUBJECT AGE', 'HR(BPM)',
       'HRC(%)', 'BELT OFF', 'LEAD OFF', 'MOTION', 'BODY POSITION']].mode().loc[0]

    File_dets={'Signal':Signal, #f[-2].split('_')[-1],
       'DevName':DevName,
       'ID':int(a['SENSOR ID']), 
       'Date':fileDate,
       'Session':Concert,
       'Interval':Work,
       'FileName':file_name,
       'FileType':'csv',
       'FileSize': fileSize,
       'RecStart':rec_start,
       'RecEnd':rec_end,
       'Duration':rec_dur,
       'BatteryStart':Batt_start,
       'BatteryEnd':Batt_end,
       'BatteryChange(mV)':Batt_spend,
       'FullLoc':eq_file_loc,}
    File_dets.update(a) # dic0.update(dic1)
    return File_dets

def matched_files(eq_file_loc,data_path,sep):
    if not sep:
        sep = '\\'
    # from the location of a good file and the location of other files, retrieve the location of all matching files
    dfile = min_dets(eq_file_loc,sep)
        # 15_RE_REIC_WW505_EQDATA.csv
    
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
            File_dets=min_dets(file,sep)
            k.append(File_dets)
    df_files=pd.DataFrame(data=k)
#     print(df_files)
    df_files.columns
    match_fields = ['ID','Date','Session','Piece','Recording']

    matched_files = df_files.loc[df_files['Recording'] == dfile['Recording']]
    for mf in match_fields[1:]:
        matched_files = matched_files.loc[matched_files[mf] == dfile[mf]]

    return list(matched_files['FullLoc'])
    
def min_dets(eq_file_loc,sep): # for csv files output by the qiosk app
    if not sep:
        sep = '\\'
    filings = eq_file_loc.split(sep)
    file_name = filings[-1]
    fileSize = os.path.getsize(eq_file_loc)
    f = file_name.split('_')
    if len(f)== 5:
        #fn_new ='_'.join([file_date,Concert,Piece,PartID,sigtype])+file_type
        # 2023-02-15_RE_REIC_WW505_EQDATA.csv
        fileDate = pd.to_datetime(f[0],format = '%Y-%m-%d')
        Session = f[1]
        Piece = f[2]
        ID = f[3]
        Signal = f[4][:-4]
        rec_name = '_'.join(f[:4])
        File_dets={'Signal':Signal, #f[-2].split('_')[-1],
           'ID':ID, 
           'Date':fileDate,
           'Session':Session,
           'Piece': Piece,
           'FileName':file_name,
           'FileSize': fileSize,
           'Recording':rec_name,
           'FullLoc':eq_file_loc}
    else:
        Signal = f[0]
        ID = f[1]#filings[-2]
        DevID = int(f[2])
        fileDate = pd.to_datetime(f[3][:6],format = '%y%m%d')
        rec_name = '_'.join(f[:4])
        # sometimes the session numbering fails and we get files with the same session number but an additiona _0 or _1
        # how to number this? What errors are producing these session numbers?
        if len(f[3].split('_'))==2: # we have an additional numbering to work into the sessions. :[
            Sessn1 = int(f[3].split('_')[0][6:8])
            Sessn2 = int(f[3].split('_')[1].split('.')[0])
            Session = (Sessn1+1)*100 + Sessn2+1 # yes this makes the session numbers huge out of nowhere, but it won't overlap with QIOSKs proper numbering that goes up to 99
        else:
            Session = int(f[3][6:8])
        File_dets={'Signal':Signal, #f[-2].split('_')[-1],
           'ID':ID, 
           'Date':fileDate,
           'Session':Session,
           'FileName':file_name,
           'FileSize': fileSize, 
           'Recording':rec_name,
           'FullLoc':eq_file_loc}
    return File_dets

