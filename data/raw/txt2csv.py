# just convert the text file to csv

import pandas as pd

df = pd.read_csv("data/raw/pd.txt", header=None)
df.to_csv("data/raw/pd.csv", index=False)