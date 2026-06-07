import pandas as pd

df = pd.read_csv('cs-training.csv')
if 'Unnamed: 0' in df.columns:
    df = df.drop('Unnamed: 0', axis=1)


print("\n--- İLK 10 SATIR ---")
print(df.head(10).to_string())