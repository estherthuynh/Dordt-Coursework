import numpy as np
import pandas as pd

df = pd.read_csv("C:\\Users\\Esther Huynh\\PycharmProjects\\cars_multi.csv")
print(df.isnull().sum())
print()
print(df["horsepower"].head(33))
# note line 32
print()
# replace null values
df["horsepower"].fillna(df.horsepower.interpolate(), inplace=True)
print()
print(df["horsepower"].head(33))
# note line 32
print()
print(df.isnull().sum())
