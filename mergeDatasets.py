import pandas as pd

train = pd.read_csv("fraudTrainEncoded.csv")
test = pd.read_csv("fraudTestEncoded.csv")

merged = pd.concat([train, test])

merged.to_csv("fraudTrainTestMerged.csv",index=False)
