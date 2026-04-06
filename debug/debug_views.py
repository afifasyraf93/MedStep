# save as debug_views.py
import pandas as pd

train = pd.read_csv("data/train_split.csv")
test  = pd.read_csv("data/test_split.csv")
full  = pd.read_csv("data/dataset_sampled.csv")

print("── View Distribution in Sampled Dataset ──")
print("\nFull sample:")
print(full["AP/PA"].value_counts())

print("\nTrain split:")
print(train["AP/PA"].value_counts())

print("\nTest split:")
print(test["AP/PA"].value_counts())

print(f"\nTotal images: {len(full)}")
print(f"PA only:  {(full['AP/PA'] == 'PA').sum()}")
print(f"AP only:  {(full['AP/PA'] == 'AP').sum()}")
print(f"Missing:  {full['AP/PA'].isna().sum()}")