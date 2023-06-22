# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 13:44:35 2023

@author: musai
"""

import pandas as pd
import sqlite3
from sklearn.metrics import classification_report
import os
import csv
import mysql.connector
from sqlalchemy import create_engine
import MySQLdb

# Connect to the database
#conn = sqlite3.connect('bsnl2')
conn =  create_engine('mysql+mysqldb://root:musu@123@localhost/bsnl2')
# Load data from database into a Pandas DataFrame
df = pd.read_sql_query("SELECT * FROM pmresult_335555397",conn)
mydb = mysql.connector.connect(
    host="localhost",
    user="root",
    password="musu@123",
    database="bsnl2"
)
 df.dropna(inplace=True)
df['Result Time'] = pd.to_datetime(df['Result Time'])

df.iloc[:,4:] = df.iloc[:,4:].apply(pd.to_numeric, errors = 'coerce')

df = df.drop(["Result Time","Granularity Period","Object Name","Reliability"],axis = 1)
df.dtypes
target_column = 'Number of Generated 404 Not Found'
train_df = df.sample(frac=0.8, random_state=42)
test_df = df.drop(train_df.index)
features = df.columns.tolist()
features.remove(target_column)

X_train = train_df[features]
y_train = train_df[target_column]
X_test = test_df[features]
y_test = test_df[target_column]

from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier()

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print(classification_report(y_test, y_pred))

df










