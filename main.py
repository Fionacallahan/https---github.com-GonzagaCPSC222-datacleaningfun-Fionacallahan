#data scientists spend the most time cleaning and organizing data 

import numpy as np
import pandas as pd 

# LOAD THE DATA 
df = pd.read_csv("pd_hoa_activities.csv")
print(df.shape) #SHAPE: (675 lines, 5 columns)

#EXPLORE THE DATA 
print(df.head(5)) #prints the first 5 rows 
print(df.tail(5)) #prints 5 rows from the end 
print("Number of participants:", df.shape[0] // 9) #9 is the number of tasks per pid 
    # this could check if some missing participants 

#print out the missing value columns 
print(df.iloc[660:670, :])

#MISSING DATA 
print(df["duration"].value_counts()["?"]) #checks how many question marks there are 
# ways to handle missing values 
# 1. discard them (if data set is big)
# 2. fill them 
#   could fill with most frequent label or central tendency meausre (median or mode)
# 3. do nothing... handle on case by case basis 
#1. let us discard and replace "?" with np.NaN (not a number)
df.replace("?", np.NaN, inplace=True) #inplace=True makes sure it replaces in memory 
print(df.isnull().sum()) #IT WORKED 
df.dropna(inplace=True) #drops the rows 
print("AFTER DROPPING:", df.isnull().sum())
print(df.shape)
#remember: now indexing will be a little messed up 
df.reset_index(inplace=True, drop=True) #fixes the indexes 
print(df.head(2))

# DECODE TASK 
# replace 1-8 and dot with more human readable/meaningful labels 
task_decoder = {"1": "Water Plants", "2": "Fill Medication Dispenser",
                "3": "Wash Countertop", "4": "Sweep and Dust",
                "5": "Cook", "6": "Wash Hands", "7": "Perform TUG",
                "8": "Perform TUG w/ Questions", "dot": "Day Out Task"}

#makes it more human readable
def decode_task(df):
    ser = df["task"]
    for key in task_decoder:
        ser.replace(key, task_decoder[key], inplace=True)
decode_task(df)
print(df.head(10))

#CLEAN CLASS 
def clean_class(df):
    ser = df["class"].copy()
    for i in range(len(ser)):
        curr_class = str(ser.iloc[i])
        curr_class = curr_class.lower()
        if "hoa" in curr_class or "healthy" in curr_class:
            ser.iloc[i] = "HOA"
        elif "pd" in curr_class or "parkinson" in curr_class:
            ser.iloc[i] = "PD"
        else:
            print("Unrecognized status: %d, %s" %(i, curr_class))
        df["class"] = ser

clean_class(df)
print(df.head(25))
print(df["class"].value_counts())

#check types of attributes 
for col in df.columns:
    print(col, df[col].dtype)
    #object: usually means string 
    #duration is stored as an object, so needs to be changed 

#print(df["duration"].mean()) #ERROR because they are strings 
df["duration"] = df["duration"].astype(np.int32)
#overwrites into an int ^
print(df["duration"].mean())
print(df["duration"].sum())
print(df["duration"].std())

# WRITE OUT THE CLEANED DATA 
df.to_csv("pd_hoa_activities_cleaned.csv", index=False)
