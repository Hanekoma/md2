import pandas as pd
from sklearn.model_selection import train_test_split

cleaned = pd.read_csv('../censusbuenodeverdad.csv')
train, test = train_test_split(cleaned, test_size=0.33)
train.to_csv('../train.csv', index=False)
test.to_csv('../test.csv', index=False)
