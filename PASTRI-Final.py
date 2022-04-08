#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 11:09:13 2022

@author: kah218
"""
#PASTRI Python Analysis Script To Retrieve Imagedata. This is written based on the output from the KREAMdonut ImageJ Macro.
import os, sys
import re, regex 
import numpy as np
import pandas as pd
import glob


#Set directories
outpath = '/Users/kah218/Desktop/CB1R Data-Filtered'
inpath_areas = '/Users/kah218/Desktop/CB1R Data-Filtered/Areas'
inpath_sd = '/Users/kah218/Desktop/CB1R Data-Filtered/StackDepth'
inpath_pa = '/Users/kah218/Desktop/CB1R Data-Filtered/ParticleAnalysis'
inpath_results = '/Users/kah218/Desktop/CB1R Data-Filtered/Results'

        
#Define functions to pull csv lists from directories. Think of it as a stack of CSVs.
def find_csv(inpath, prefix):
    csvlist = []
    for f in glob.glob(inpath + '/'+prefix+'*.csv'):
        temp_df = pd.read_csv(f)
        csvlist.append(temp_df)
    return(csvlist)

def find_csv_sd(inpath, prefix):
    csvlist = []
    for f in glob.glob(inpath + '/'+prefix+'*.csv'):
        temp_df = pd.read_csv(f, header=None, names=["Data"])
        temp_df["Filename"] = pd.Series(os.path.basename(f)) 
        csvlist.append(temp_df)
    return(csvlist)

#
#https://stackoverflow.com/questions/42756696/read-multiple-csv-files-and-add-filename-as-new-column-in-pandas
'''
#The above function isn't recursive. Here is an alternative that is.
def find_csv2(csv_dir, prefix):
    csvlist = []
    for root, dirs, names in os.walk(csv_dir):
        for file in names:
            # print(file)
            if file.startswith(prefix):
                f = os.path.join(root, file)
                # print(f)
                csvlist.append(f)
    return(csvlist)
'''


#Define functions to concatenate data. This part esspentially combines that stack of CSVs into one sheet.
def concat_data(csvlist):
    df = pd.concat(csvlist)
    return(df)


#Extract Basename using regex
def basename(df, Filename):
    df=df.rename(columns={Filename:'Filename'})
    tempdf = []
    for i, row in df.iterrows():
        tempdf.append((str)((pd.Series((re.search(r'BR(M|F)1?[0-9]-vHPC-[1-4]-u?1?[0-9]',(str)(row.Filename))).group(0))).values[0]))
    df=df.assign(Extract=tempdf)
    return(df)


#Extract Background Basename using regex
def background_basename(df, Filename):
    df=df.rename(columns={Filename:'Filename'})
    tempdf = []
    for i, row in df.iterrows():
        tempdf.append((str)((pd.Series((re.search(r'BR(M|F)1?[0-9]-vHPC-[1-4]-background',(str)(row.Filename))).group(0))).values[0]))
    df=df.assign(Extract=tempdf)
    return(df)


#Function to normalize data (I use this to normalize by the Volume)
def norm_df(df1, df2):
    #  df1: The dataframe to be operated on
    #  df2: the dataframe to divide df1 by (assumes a single column of values)
    #  out_df: the output dataframe
    out_df = df1.copy(deep=True) ## 2022.03.24 --SGT -- Makes a copy of df1 rather than just pointing to df1 (which is what out_df = df1 would do)
    for col in out_df.columns.values:
        out_df.loc[:,col] = out_df.loc[:,col].div(df2.squeeze(), axis='index')  ## 2022.03.24 -- SGT -- Adjusted this to ensure df2 had indices to match to df1/out_df
    out_df.set_index(df1.index,inplace=True)
    return out_df


#Function to subtract data (I use this to subtract the normalized Background data).
def sub_df(df1, df2):
    df_out = pd.DataFrame()
    for i, row in df2.iterrows():
        sec_str = (str)(row.name).replace("-background","")
        df_out = pd.concat([df_out, df1.filter(like=sec_str, axis=0).sub(row.values, axis=1)])
    return df_out


#Stack Depth Compilation 
csvlist_sd = find_csv_sd(inpath_sd, "")

df_sd_full = concat_data(csvlist_sd)

tempdf = pd.Series(dtype=str)
for row in df_sd_full.itertuples():
    temps =(re.search(r'BR(M|F)1?[0-9]-vHPC-[1-4]-u?1?[0-9]',(str)(row.Filename)))
    if temps == None:
        temps = (re.search(r'BR(M|F)1?[0-9]-vHPC-[1-4]-background',(str)(row.Filename)))
    tempdf = pd.concat([tempdf,pd.Series(temps.group(0))])

df_sd_full = pd.concat([df_sd_full, tempdf.rename('Extract')], axis=1)

df_sd = df_sd_full[df_sd_full['Extract'].str.contains(r'BR(M|F)1?[0-9]-vHPC-[1-4]-u?1?[0-9]', case=False, regex=True)]

df_backsd = df_sd_full[df_sd_full['Extract'].str.contains(r'BR(M|F)1?[0-9]-vHPC-[1-4]-background', case=False, regex=True)]

df_sd.set_index('Extract', inplace=True)
df_backsd.set_index('Extract', inplace=True)


#All 3 ROI Area Data Compilation for cell data as well as for background.
csvlist_areas = find_csv(inpath_areas, "")

df_areas = concat_data(csvlist_areas)

df_areas = df_areas[df_areas['Label'].str.contains(r'BR(M|F)1?[0-9]-vHPC-[1-4]-u?1?[0-9]', case=False, regex=True)]
df_inner = df_areas[df_areas['Label'].str.contains("inner", case=False, regex=False)]
df_outer = df_areas[df_areas['Label'].str.contains("outer", case=False, regex=False)]
df_donut = df_areas[df_areas['Label'].str.contains("donut", case=False, regex=False)]

df_donut = basename(df_donut,'Label')
df_donut.set_index('Extract', inplace=True)

df_backareas = concat_data(csvlist_areas)
df_backareas = df_backareas[df_backareas['Label'].str.contains(r'BR(M|F)1?[0-9]-vHPC-[1-4]-background', case=False, regex=True)]
df_backinner = df_backareas[df_backareas['Label'].str.contains("inner", case=False, regex=False)]
df_backouter = df_backareas[df_backareas['Label'].str.contains("outer", case=False, regex=False)]
df_backdonut = df_backareas[df_backareas['Label'].str.contains("donut", case=False, regex=False)]

df_backdonut = background_basename(df_backdonut,'Label')
df_backdonut.set_index('Extract',inplace=True)

'''
#Calculate volume from Area & Stack Depth. This will be used to normalize the data.
df_sd.Data *= 0.33 #Multiply the number of images in the stack by the known step size (units here: um)
df_volume = df_donut[['Area']].multiply(df_sd['Data'], axis="index")
df_volume.rename(columns={'Area':'Volume'}, inplace=True)

df_backsd.Data *= 0.33
df_backvolume = df_backdonut[['Area']].multiply(df_backsd['Data'], axis="index")
df_backvolume.rename(columns={'Area':'Volume'}, inplace=True)
'''

#Intensity Results Compilation for cell data and background
csvlist_intensity = find_csv(inpath_results, "")
df_intensity_full = concat_data(csvlist_intensity)
df_intensity = df_intensity_full[df_intensity_full['Label'].str.contains(r'BR(M|F)1?[0-9]-vHPC-[1-4]-u?1?[0-9]', case=False, regex=True)]
df_backint = df_intensity_full[df_intensity_full['Label'].str.contains(r'BR(M|F)1?[0-9]-vHPC-[1-4]-background', case=False, regex=True)]

#Summary Particle Analysis Compilation (cell data only)
csvlist_pa = find_csv(inpath_pa, "Summary")

df_pa = concat_data(csvlist_pa)
df_pa = df_pa[df_pa['Slice'].str.contains(r'BR(M|F)1?[0-9]-vHPC-[1-4]-u?1?[0-9]', case=False, regex=True)]

'''
#Normalize Results and Summary Particle Analysis Data by Volume.
#First have to rename the animal labels in the index to get them to match Stack Depth. 
#(resource: https://regexr.com and https://stackoverflow.com/questions/62029446/python-regex-extract-date-to-new-column-in-dataframe and https://pypi.org/project/regex/)
'''
df_pa = basename(df_pa,'Slice')
df_pa.set_index('Extract', inplace=True)

del df_pa['Filename']

df_intensity = basename(df_intensity,'Label')
df_intensity.set_index('Extract', inplace=True)

del df_intensity['Filename']
del df_intensity[' ']

df_backint = background_basename(df_backint,'Label')
df_backint.set_index('Extract', inplace=True)

del df_backint['Filename']
del df_backint[' ']
'''
df_normintensity = norm_df(df_intensity, df_volume)

df_normpa = norm_df(df_pa, df_volume)

df_normparea = norm_df(df_pa, df_sd[['Data']])

df_normbackground = norm_df(df_backint, df_backvolume)

#Subtract background intensity from Results intensity measurements after normalization by Volume
df_subnormint = sub_df(df_normintensity, df_normbackground)
'''

#Compile Individual Particle Analysis Data. Not doing much processing of this as I don't expect us to use it much except to check.
csvlist_ind = find_csv(inpath_pa, "Results")

with pd.ExcelWriter(outpath+"/IndPAData.xlsx", engine='xlsxwriter') as writer:
    for csv in csvlist_ind:
        temps =(re.search(r'BR(M|F)1?[0-9]-vHPC-[1-4]-u?1?[0-9]',(str)(csv.Label.values[0])))
        if temps == None:
            temps = (re.search(r'BR(M|F)1?[0-9]-vHPC-[1-4]-background',(str)(csv.Label.values[0])))
        csv.to_excel(writer, sheet_name=temps.group(0)) 
    writer.save()
  
    
#Split the ROI Area and Stack Depth Data & Save as an Excel Sheet
df_inner = basename(df_inner, 'Label')
df_inner.reset_index(inplace=True)
df_inner[['Animal', 'Subregion', 'Section', 'Cell']] = df_inner['Extract'].astype("string").str.split('-', expand=True)
df_inner.set_index('Animal')
df_inner['Labeled'] = np.where(df_inner['Cell'].str.contains("u"), 'Unlabeled', 'Labeled')

df_outer = basename(df_outer, 'Label')
df_outer.reset_index(inplace=True)
df_outer[['Animal', 'Subregion', 'Section', 'Cell']] = df_outer['Extract'].astype("string").str.split('-', expand=True)
df_outer.set_index('Animal')
df_outer['Labeled'] = np.where(df_outer['Cell'].str.contains("u"), 'Unlabeled', 'Labeled')

df_donut.reset_index(inplace=True)
df_donut[['Animal', 'Subregion', 'Section', 'Cell']] = df_donut['Extract'].astype("string").str.split('-', expand=True)
df_donut.set_index('Animal')
df_donut['Labeled'] = np.where(df_donut['Cell'].str.contains("u"), 'Unlabeled', 'Labeled')

df_sd.reset_index(inplace=True)
df_sd[['Animal', 'Subregion', 'Section', 'Cell']] = df_sd['Extract'].astype("string").str.split('-', expand=True)
df_sd.set_index('Animal', inplace=True)
df_sd['Labeled'] = np.where(df_sd['Cell'].str.contains("u"), 'Unlabeled', 'Labeled')
'''
df_volume.reset_index(inplace=True)
df_volume[['Animal', 'Subregion', 'Section', 'Cell']] = df_volume['Extract'].astype("string").str.split('-', expand=True)
df_volume.set_index('Animal')
df_volume['Labeled'] = np.where(df_volume['Cell'].str.contains("u"), 'Unlabeled', 'Labeled')
'''
df_backinner = background_basename(df_backinner, 'Label')
df_backinner.reset_index(inplace=True)
df_backinner[['Animal', 'Subregion', 'Section', 'Cell']] = df_backinner['Extract'].astype("string").str.split('-', expand=True)
df_backinner.set_index('Animal')
df_backinner['Labeled'] = np.where(df_backinner['Cell'].str.contains("background"), 'Background', '0')

df_backouter = background_basename(df_backouter, 'Label')
df_backouter.reset_index(inplace=True)
df_backouter[['Animal', 'Subregion', 'Section', 'Cell']] = df_backouter['Extract'].astype("string").str.split('-', expand=True)
df_backouter.set_index('Animal')
df_backouter['Labeled'] = np.where(df_backouter['Cell'].str.contains("background"), 'Background', '0')

df_backdonut.reset_index(inplace=True)
df_backdonut[['Animal', 'Subregion', 'Section', 'Cell']] = df_backdonut['Extract'].astype("string").str.split('-', expand=True)
df_backdonut.set_index('Animal')
df_backdonut['Labeled'] = np.where(df_backdonut['Cell'].str.contains("background"), 'Background', '0')

df_backsd.reset_index(inplace=True)
df_backsd[['Animal', 'Subregion', 'Section', 'Cell']] = df_backsd['Extract'].astype("string").str.split('-', expand=True)
df_backsd.set_index('Animal')
df_backsd['Labeled'] = np.where(df_backsd['Cell'].str.contains("background"), 'Background', '0')
'''
df_backvolume.reset_index(inplace=True)
df_backvolume[['Animal', 'Subregion', 'Section', 'Cell']] = df_backvolume['Extract'].astype("string").str.split('-', expand=True)
df_backvolume.set_index('Animal')
df_backvolume['Labeled'] = np.where(df_backvolume['Cell'].str.contains("background"), 'Background', '0')
'''
with pd.ExcelWriter(outpath+"/AreasAndVolume.xlsx", engine='xlsxwriter') as writer:
    df_inner.to_excel(writer, sheet_name="Inner")
    df_outer.to_excel(writer, sheet_name="Outer")
    df_donut.to_excel(writer, sheet_name="Donut")
    df_sd.to_excel(writer, sheet_name="Stack_Depth")
    #df_volume.to_excel(writer, sheet_name="Volume")
    df_backinner.to_excel(writer, sheet_name="Background_Inner")
    df_backouter.to_excel(writer, sheet_name="Background_Outer")
    df_backdonut.to_excel(writer, sheet_name="Background_Donut")
    df_backsd.to_excel(writer, sheet_name="Background_Stack_Depth")
    #df_backvolume.to_excel(writer, sheet_name="Background_Volume")
    writer.save()


#Per Mikaela's Suggestion
df_subInt = sub_df(df_intensity,df_backint)

#Split and save all summary df to Single Excel
'''
df_subnormint.reset_index(inplace=True)
df_subnormint[['Animal', 'Subregion', 'Section', 'Cell']] = df_subnormint['Extract'].astype("string").str.split('-', expand=True)
df_subnormint.set_index('Animal')
df_subnormint['Labeled'] = np.where(df_subnormint['Cell'].str.contains("u"), 'Unlabeled', 'Labeled')
'''
df_subInt.reset_index(inplace=True)
df_subInt[['Animal', 'Subregion', 'Section', 'Cell']] = df_subInt['Extract'].astype("string").str.split('-', expand=True)
df_subInt.set_index('Animal')
df_subInt['Labeled'] = np.where(df_subInt['Cell'].str.contains("u"), 'Unlabeled', 'Labeled')
'''
df_normparea.reset_index(inplace=True)
df_normparea[['Animal', 'Subregion', 'Section', 'Cell']] = df_normparea['Extract'].astype("string").str.split('-', expand=True)
df_normparea.set_index('Animal')
df_normparea['Labeled'] = np.where(df_normparea['Cell'].str.contains("u"), 'Unlabeled', 'Labeled')

df_normpa.reset_index(inplace=True)
df_normpa[['Animal', 'Subregion', 'Section', 'Cell']] = df_normpa['Extract'].astype("string").str.split('-', expand=True)
df_normpa.set_index('Animal')
df_normpa['Labeled'] = np.where(df_normpa['Cell'].str.contains("u"), 'Unlabeled', 'Labeled')

df_normintensity.reset_index(inplace=True)
df_normintensity[['Animal', 'Subregion', 'Section', 'Cell']] = df_normintensity['Extract'].astype("string").str.split('-', expand=True)
df_normintensity.set_index('Animal', inplace=True)
df_normintensity['Labeled'] = np.where(df_normintensity['Cell'].str.contains("u"), 'Unlabeled', 'Labeled')

df_normbackground.reset_index(inplace=True)
df_normbackground[['Animal', 'Subregion', 'Section', 'Cell']] = df_normbackground['Extract'].astype("string").str.split('-', expand=True)
df_normbackground.set_index('Animal', inplace=True)
df_normbackground['Labeled'] = np.where(df_normbackground['Cell'].str.contains("background"), 'Background', '0')
'''
df_intensity.reset_index(inplace=True)
df_intensity[['Animal', 'Subregion', 'Section', 'Cell']] = df_intensity['Extract'].astype("string").str.split('-', expand=True)
df_intensity.set_index('Animal', inplace=True)
df_intensity['Labeled'] = np.where(df_intensity['Cell'].str.contains("u"), 'Unlabeled', 'Labeled')

df_pa.reset_index(inplace=True)
df_pa[['Animal', 'Subregion', 'Section', 'Cell']] = df_pa['Extract'].astype("string").str.split('-', expand=True)
df_pa.set_index('Animal', inplace=True)
df_pa['Labeled'] = np.where(df_pa['Cell'].str.contains("u"), 'Unlabeled', 'Labeled')

with pd.ExcelWriter(outpath+"/Data.xlsx", engine='xlsxwriter') as writer:
    #df_subnormint.to_excel(writer, sheet_name="Subtracted_NormIntensity")
    #df_normparea.to_excel(writer, sheet_name="Normalized_Percent_Area")
    #df_normpa.to_excel(writer, sheet_name="Normalized_Particle_Analysis")
    #df_normintensity.to_excel(writer, sheet_name="NormIntPreSub")
    #df_normbackground.to_excel(writer, sheet_name="NormBackInt")
    df_subInt.to_excel(writer, sheet_name="Subtracted_Intensity")
    df_intensity.to_excel(writer, sheet_name="Int")
    df_pa.to_excel(writer, sheet_name="Particle_Analysis")
    writer.save()


#Focus on Specific Data and Collapse Cells
df_subInt.reset_index(inplace=True)
df_subInt[['Animal', 'Subregion', 'Section', 'Cell']] = df_subInt['Extract'].astype("string").str.split('-', expand=True)
df_subInt.set_index('Animal')
df_subInt['Labeled'] = np.where(df_subInt['Cell'].str.contains("u"), 'Unlabeled', 'Labeled')
df_SubInt_IntDen = df_subInt.groupby(['Animal', 'Section', 'Labeled'])['IntDen'].mean()
df_SubInt_IntDen2 = df_SubInt_IntDen.unstack(level=-1)

df_SubInt_Mean = df_subInt.groupby(['Animal', 'Section', 'Labeled'])['Mean'].mean()
df_SubInt_Mean2 = df_SubInt_Mean.unstack(level=-1)

#df_SubInt_Max = df_subInt.groupby(['Animal', 'Section', 'Labeled'])['Max'].mean()
#df_SubInt_Max2 = df_SubInt_Max.unstack(level=-1)
'''
df_SubAvgInt = df_subnormint.groupby(['Animal', 'Section', 'Labeled'])['Mean'].mean()
df_SubAvgInt2 = df_SubAvgInt.unstack(level=-1)

df_SubMaxInt = df_subnormint.groupby(['Animal', 'Section', 'Labeled'])['Max'].mean()
df_SubMaxInt2 = df_SubMaxInt.unstack(level=-1)

df_NormIntPreSub_Avg =  df_normintensity.groupby(['Animal', 'Section', 'Labeled'])['Mean'].mean()
df_NormIntPreSub_Avg2 = df_NormIntPreSub_Avg.unstack(level=-1)

df_NormIntPreSub_Max = df_normintensity.groupby(['Animal', 'Section', 'Labeled'])['Max'].mean()
df_NormIntPreSub_Max2 = df_NormIntPreSub_Max.unstack(level=-1)

df_PerArea = df_normparea.groupby(['Animal', 'Section', 'Labeled'])['%Area'].mean()
df_PerArea2 = df_PerArea.unstack(level=-1)
'''
df_PA = df_pa.groupby(['Animal', 'Section', 'Labeled'])['%Area'].mean()
df_PA2 = df_PA.unstack(level=-1)

with pd.ExcelWriter(outpath+"/DataForPrism.xlsx", engine='xlsxwriter') as writer:
    df_SubInt_IntDen2.to_excel(writer, sheet_name="Subtracted_Integrated_Density")
    df_SubInt_Mean2.to_excel(writer, sheet_name="Subtracted_Intensity_Mean")
    #df_SubInt_Max2.to_excel(writer, sheet_name="Subtracted-Intensity_Max")
    #df_SubAvgInt2.to_excel(writer, sheet_name="Subtracted_NormIntensity_Mean")
    #df_SubMaxInt2.to_excel(writer, sheet_name="Subtracted_NormIntensity_Max")
    #df_NormIntPreSub_Avg2.to_excel(writer, sheet_name="Non-Substracted_NormInt_Mean")
    #df_NormIntPreSub_Max2.to_excel(writer, sheet_name="Non-Subtracted_NormInt_Max")
    df_PA2.to_excel(writer, sheet_name="Percent Area")
    #df_PerArea2.to_excel(writer, sheet_name="Normalized Percent Area from PA")
    writer.save()