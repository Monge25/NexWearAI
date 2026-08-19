import pandas as pd

df = pd.read_csv("training/discount_dataset.csv")
print(df.head())
print("\nValores nulos por columna:\n", df.isnull().sum())
print("\nTipos de dato:\n", df.dtypes)