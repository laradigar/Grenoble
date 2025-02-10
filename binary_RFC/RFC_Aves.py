#%% Import modules and libraries
import torch
import torchaudio
import pandas as pd
import os
import re
from torch.utils.data import Dataset, DataLoader
import torchaudio.transforms as T
import torch.nn as nn
import librosa
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import joblib
import random

#%% Read CSV file with filenames and groups and make it appropriate as an annotation file
#groups = ["Anthropophony", "Aves", "Geophony", "Mammalia", "Orthoptera", "Hemiptera", "Diptera", "Hymenoptera", "Anura", "Insecta"]

embeds = pd.read_csv('/Users/laradiazgarcia/Desktop/Grenoble/embbedings_with_groups.csv')
data = pd.DataFrame()
data["filename"] = embeds["Filename"]
data = pd.concat([data, embeds.iloc[:, 2:514]], axis=1)
# Convert group column (comma-separated) into lists and check if 'Anthropophony' is present
data['is_aves'] = embeds["group"].str.contains('Aves').astype(int)
# Save to CSV
data.to_csv('is_aves.csv', index=False)

# Print results
print(data.head())

#%% New way to divide the files to ensure 70/30 and 75/25 splits
df = pd.read_csv("/Users/laradiazgarcia/Desktop/Grenoble/binary_RFC/is_aves.csv")

# Group files by prefix
file_groups = {}  # {prefix: [list of files]}
for f in df["filename"]:
    prefix = re.sub(r"_split_\d+_\d+\.wav$", "", f)  # Extract prefix
    if prefix not in file_groups:
        file_groups[prefix] = []
    file_groups[prefix].append(f)

# Assign labels to prefixes (use the first file in the group)
prefix_labels = {}  # {prefix: label}
for prefix, files in file_groups.items():
    label = df[df["filename"] == files[0]]["is_aves"].values[0]  # Get label
    prefix_labels[prefix] = label

# Split prefixes into two lists based on their labels
prefixes_0 = [p for p in prefix_labels if prefix_labels[p] == 0]  # Class 0
prefixes_1 = [p for p in prefix_labels if prefix_labels[p] == 1]  # Class 1

# Shuffle groups randomly
random.seed(42)
random.shuffle(prefixes_0)
random.shuffle(prefixes_1)

# Target total size
total_target_size = 4453 + random.randint(-40, 40) # Approximate total count + random factor to consider different datasets
train_target_size = int(total_target_size * 0.7)  # 70% Train
valid_target_size = total_target_size - train_target_size  # 30% Validation

# Compute how many of each class we need
train_0_target = int(train_target_size * 0.75)  # 75% of training set
train_1_target = train_target_size - train_0_target  # 25% of training set
valid_0_target = int(valid_target_size * 0.75)  # 75% of validation set
valid_1_target = valid_target_size - valid_0_target  # 25% of validation set

# Assign prefixes to train and validation sets
train_prefixes_0 = []
train_prefixes_1 = []
valid_prefixes_0 = []
valid_prefixes_1 = []

# Track the number of files assigned
train_0_count, train_1_count = 0, 0
valid_0_count, valid_1_count = 0, 0

# Assign prefixes while counting files
for prefix in prefixes_0:
    if train_0_count + len(file_groups[prefix]) <= train_0_target:
        train_prefixes_0.append(prefix)
        train_0_count += len(file_groups[prefix])
    elif valid_0_count + len(file_groups[prefix]) <= valid_0_target:
        valid_prefixes_0.append(prefix)
        valid_0_count += len(file_groups[prefix])

for prefix in prefixes_1:
    if train_1_count + len(file_groups[prefix]) <= train_1_target:
        train_prefixes_1.append(prefix)
        train_1_count += len(file_groups[prefix])
    elif valid_1_count + len(file_groups[prefix]) <= valid_1_target:
        valid_prefixes_1.append(prefix)
        valid_1_count += len(file_groups[prefix])

# Combine sets
train_prefixes = set(train_prefixes_0 + train_prefixes_1)
valid_prefixes = set(valid_prefixes_0 + valid_prefixes_1)

# Get all files in each dataset
train_files = [f for prefix in train_prefixes for f in file_groups[prefix]]
valid_files = [f for prefix in valid_prefixes for f in file_groups[prefix]]

# Create final dataframes
df_train = df[df["filename"].isin(train_files)]
df_valid = df[df["filename"].isin(valid_files)]

# Save to CSV
df_train.to_csv("train_annotations_aves.csv", index=False)
df_valid.to_csv("valid_annotations_aves.csv", index=False)

# Print stats
num_train_0 = sum(df_train["is_aves"] == 0)
num_train_1 = sum(df_train["is_aves"] == 1)
num_valid_0 = sum(df_valid["is_aves"] == 0)
num_valid_1 = sum(df_valid["is_aves"] == 1)

print(f"Train set: {len(train_files)} files ({num_train_0} class 0, {num_train_1} class 1)")
print(f"Validation set: {len(valid_files)} files ({num_valid_0} class 0, {num_valid_1} class 1)")


#%% Doing a Random Forest model 
train_df_with_features = pd.read_csv('train_annotations_aves.csv')
valid_df_with_features = pd.read_csv('valid_annotations_aves.csv') 

# Check that there is no overlap in the file names between datasets
def extract_prefix(filename):
    return "_".join(filename.split("_")[:-3])  # Remove everything after the last 2 underscores

# Create sets of prefixes
train_prefixes = set(train_df_with_features["filename"].apply(extract_prefix))
valid_prefixes = set(valid_df_with_features["filename"].apply(extract_prefix))

# Find overlaps
overlapping_prefixes = train_prefixes.intersection(valid_prefixes)

# Print results
if overlapping_prefixes:
    print(f"❌ Overlapping prefixes found: {len(overlapping_prefixes)}")
    print(overlapping_prefixes)  # Print the overlapping prefixes
else:
    print("✅ No overlapping prefixes between train and validation sets!")

# Drop filename column and split into X and y
X_train = train_df_with_features.drop(columns=["filename", "is_aves"])
y_train = train_df_with_features["is_aves"]
X_valid = valid_df_with_features.drop(columns=["filename", "is_aves"])
y_valid = valid_df_with_features["is_aves"]

# Initialize and train the RandomForest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)  # 100 trees
rf_model.fit(X_train, y_train)

# Make predictions on the validation set
y_pred = rf_model.predict(X_valid)

# Evaluate model performance
accuracy = accuracy_score(y_valid, y_pred)
print(f"Validation Accuracy: {accuracy:.4f}")

# Print a full classification report
print(classification_report(y_valid, y_pred))

# Write the file distribution, accuracy, and classification report to a .txt
with open('RFC_accuracy_aves.txt', 'w') as f:
    f.write(f"Total files used: {len(train_files)+len(valid_files)}\n")
    f.write(f"Train: {len(train_files)} ({num_train_1} '1' labels, {num_train_0} '0' labels)\n")
    f.write(f"Validation: {len(valid_files)} ({num_valid_1} '1' labels, {num_valid_0} '0' labels)\n")
    f.write(f"Validation Accuracy: {accuracy:.4f}\n")
    f.write(classification_report(y_valid, y_pred))
print("Classification report saved to RFC_accuracy_aves.txt")

#%% Classification and prediction at whole recording level
predicts = pd.DataFrame({"filename": valid_df_with_features.filename, "y_true": y_valid, "y_pred": y_pred})
predicts.to_csv('predictions_aves.csv', index=False)

# Load predictions and ground truth
df = pd.read_csv("predictions_aves.csv")  # Ensure it contains: filename, y_true, y_pred

# Function to extract prefix from filename
def extract_prefix(filename):
    return "_".join(filename.split("_")[:-3])  # Remove everything after the last 2 underscores

# Add a prefix column
df["prefix"] = df["filename"].apply(extract_prefix)

# Aggregate predictions: If any segment is predicted as 1, the whole prefix is 1
grouped_predictions = df.groupby("prefix")["y_pred"].max().reset_index()
grouped_truth = df.groupby("prefix")["y_true"].max().reset_index()

# Merge grouped data to compare predictions vs. true labels
final_eval_df = grouped_truth.merge(grouped_predictions, on="prefix")

# Compute evaluation metrics
print("Classification Report (Prefix-Level Evaluation):")
accuracy = accuracy_score(final_eval_df["y_true"], final_eval_df["y_pred"])
print(f"Validation Accuracy: {accuracy:.4f}")
print(classification_report(final_eval_df["y_true"], final_eval_df["y_pred"]))

with open('RFC_accuracy_wholeaudio_aves.txt', 'w') as f:
    f.write("(Prefix-Level Evaluation) \n")
    f.write(f"Validation Accuracy: {accuracy:.4f}\n")
    f.write(classification_report(final_eval_df["y_true"], final_eval_df["y_pred"]))
print("Classification report saved to RFC_accuracy_wholeaudio_aves.txt")
