import numpy as np
import pandas as pd

# Load data
titanic = pd.read_csv("./data/creditcard.csv")
print(titanic.head(5))

# Split fraud and genuine transactions
df = titanic.copy()
fraud = df.loc[df["Class"] == 1]
fraud = fraud.reset_index()
df = titanic.copy()
genuine = df.loc[df["Class"] == 0]
genuine = genuine.reset_index()

# keep 2000 genuine transactions
genuine_1 = genuine[0:1000]
genuine_2 = genuine[1000:2000]

# insert frauds in genuine transactions
fraud_in_genuine = pd.concat([genuine_1, fraud, genuine_2]).reset_index()
# fraud_in_genuine = fraud_in_genuine[["V1", "V2", "V3", "V4", "V5"]]
fraud_in_genuine = fraud_in_genuine[["V1", "V2", "V3", "V4"]]
Y = fraud_in_genuine.values

np.save('./baselines/data/creditcard.npy', Y.transpose())