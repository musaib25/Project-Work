# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 13:38:02 2023

@author: musai
"""

import os
import pandas as pd
import csv
import mysql.connector
import sqlite3
from sqlalchemy import create_engine
import MySQLdb

mydb = mysql.connector.connect(
    host="localhost",
    user="root",
    password="xxxxxxxxx",
    database="xxxxx"
)
mycursor = mydb.cursor()
k=1

conn =  create_engine('mysql+mysqldb://root:xxxxxxxxx@localhost/xxxxx')
table_name="pmresult_335555339"
oby="Result Time"
src_path_days= "Your required path"
list_of_days = os.listdir(src_path_days)


for days in list_of_days:
    src_path_hours="Your Required path " + str(days)
    list_of_hours=os.listdir(src_path_hours)

    for hours in list_of_hours:
        os.chdir(src_path_hours)
        if hours.startswith("pmresult_335555339"):
            df2=pd.read_csv(hours)
            New = df2.drop(labels=[0])
            New_Data = New.reset_index(drop=True)    
            New_Data = New_Data.rename(columns=lambda x:x[:46])
            column_names = list(New_Data.columns)
            New_Data.columns = New_Data.columns.str.strip()

            if k==1:
                conn =  create_engine('mysql+mysqldb://root:xxxxxxxxx@localhost/xxxxx')
                New_Data.to_sql(table_name, conn, if_exists='replace', index=False)
                sql_create_table = f"CREATE TABLE {table_name} ({','.join(column_names)})"
                k=0
            else:
                New_Data.to_sql(table_name, conn, if_exists='append', index=False)

import mysql.connector as connection


try:
    mydb = mysql.connector.connect(host="localhost", database = 'xxxxx',user="root", passwd="xxxxxxxxx",use_pure=True)
    query = "Select * from pmresult_335555339;"
    result_dataFrame = pd.read_sql(query,mydb)
    mydb.close() #close the connection
except Exception as e:
    mydb.close()
    print(str(e))
    result_dataFrame = pd.read_sql(query, mydb)
result_dataFrame.head()
print(result_dataFrame)

import seaborn as sns
Retreived =result_dataFrame.iloc[:, 5:]

Convert =  Retreived.astype(float)
f = Convert.astype('int')
df = f.loc[:, f.sum(axis=0) > 0]

f = f.select_dtypes(exclude=['object'])
df = df.loc[:, df.sum(axis=0) > 0]
corr_matrix = df.corr()


import numpy as np
import matplotlib.pyplot as plt

corr_matrix = df.corr()

threshold = 0.90

mask = (corr_matrix) >= threshold

corr_cols = set()
for i in range(len(mask.columns)):
    for j in range(i):
        if mask.iloc[i,j]:
            corr_cols.add(mask.columns[i])
            corr_cols.add(mask.columns[j])

print(corr_cols)

upper = corr_matrix.where(
    np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

to_drop = [column for column in upper.columns if any(upper[column] < 0.9)]
to_drop

# Find index of feature columns with correlation equal to 1
perft_Corr = [column for column in upper.columns if any(upper[column] == 1)]
perft_Corr

# Find index of feature columns with correlation greater than 0.95
pos = [column for column in upper.columns if any(upper[column] > 0.95)]
pos

new_df = df[to_drop]
new_df
sns.pairplot(new_df, kind='reg', diag_kind='kde')

corr = corr_matrix = df.corr()

plt.figure(figsize=(30, 10))
sns.heatmap(corr_matrix, cmap='coolwarm', annot=True, mask=False)
plt.show()





