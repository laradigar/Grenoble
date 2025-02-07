
"""
This script performs the following tasks:
1. Imports necessary modules and libraries.
2. Reads a CSV file containing filenames and groups, processes it to create an annotation file indicating the presence of 'Orthoptera'.
3. Splits the dataset into training and validation sets based on unique file prefixes.
4. Extracts Mel-frequency cepstral coefficients (MFCC) features from audio files for both training and validation sets.
5. Trains a Random Forest model using the extracted MFCC features and evaluates its performance on the validation set.

Functions:
- extract_features(file_path): Extracts MFCC features from an audio file.
- add_mfcc_features(df): Adds MFCC features to a given dataframe containing filenames.

Data Files:
- embbedings_with_groups.csv: Input CSV file with filenames and groups.
- is_orthoptera.csv: Output CSV file indicating the presence of 'Orthoptera'.
- train_annotations.csv: Output CSV file with training data annotations.
- valid_annotations.csv: Output CSV file with validation data annotations.
- train_with_mfcc.csv: Output CSV file with training data and MFCC features.
- valid_with_mfcc.csv: Output CSV file with validation data and MFCC features.
"""
#%% Import modules and libraries
import torch
import torchaudio
import pandas as pd
import os
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

embeds = pd.read_csv('/Users/laradiazgarcia/Desktop/Grenoble/embbedings_with_groups.csv')
# Check if "Orthoptera" is in the group column
data = pd.DataFrame()
data["filename"] = embeds["Filename"]
data = pd.concat([data, embeds.iloc[:, 2:514]], axis=1)
# Convert group column (comma-separated) into lists and check if 'Orthoptera' is present
data['is_orthoptera'] = embeds["group"].str.contains('Orthoptera').astype(int)
# Save to CSV
data.to_csv('is_orthoptera.csv', index=False)

# Print results
print(data.head())

#%% Determine which of my files are training data and which are validation data
random.seed(42)
folder_path = "/Users/laradiazgarcia/Desktop/Grenoble/All"
file_names = [f for f in os.listdir(folder_path) if f.endswith('.wav')]
df = pd.read_csv("/Users/laradiazgarcia/Desktop/Grenoble/binary_RFC/is_orthoptera.csv")

# Extract unique prefixes (everything before beginningtime_endtime)
file_groups = {}  # {prefix: [list of files]}
for f in file_names:
    prefix = "_".join(f.split("_")[:-3])  # Remove last two parts (beginningtime and endtime)
    if prefix not in file_groups:
        file_groups[prefix] = []
    file_groups[prefix].append(f)

# Assign labels to each prefix (based on first file in group)
prefix_labels = {}
for prefix, files in file_groups.items():
    first_file = files[0]  # Pick any file from the group to check its label
    label = df[df['filename'] == first_file]['is_orthoptera'].values
    if len(label) > 0:
        prefix_labels[prefix] = label[0]  # Store label for the whole group

# Separate prefixes by label
prefixes_0 = [p for p in prefix_labels if prefix_labels[p] == 0]
prefixes_1 = [p for p in prefix_labels if prefix_labels[p] == 1]

# Shuffle prefixes
random.shuffle(prefixes_0)
random.shuffle(prefixes_1)

# Define total desired split
total_train_size = 0.7  # 70% total dataset should be training
total_valid_size = 0.3  # 30% total dataset should be validation

# Define class proportions within each split
train_1_ratio = 0.25  # 25% of training should be label 1
train_0_ratio = 0.75  # 75% of training should be label 0
valid_1_ratio = 0.25  # 25% of validation should be label 1
valid_0_ratio = 0.75  # 75% of validation should be label 0

# Compute how many 1s and 0s should be in each set
num_total = len(prefix_labels)
num_train = int(num_total * total_train_size)
num_valid = num_total - num_train  # Remaining goes to validation

num_train_1 = int(num_train * train_1_ratio)
num_train_0 = num_train - num_train_1

num_valid_1 = int(num_valid * valid_1_ratio)
num_valid_0 = num_valid - num_valid_1

# Assign the correct number of prefixes for each label
train_prefixes_1 = set(prefixes_1[:num_train_1])
train_prefixes_0 = set(prefixes_0[:num_train_0])
train_prefixes = train_prefixes_1.union(train_prefixes_0)

valid_prefixes_1 = set(prefixes_1[num_train_1:num_train_1 + num_valid_1])
valid_prefixes_0 = set(prefixes_0[num_train_0:num_train_0 + num_valid_0])
valid_prefixes = valid_prefixes_1.union(valid_prefixes_0)

# Assign files to train/validation
train_files = [f for prefix in train_prefixes for f in file_groups[prefix]]
valid_files = [f for prefix in valid_prefixes for f in file_groups[prefix]]

# Filter annotations for train/validation
df_train = df[df['filename'].isin(train_files)]
df_valid = df[df['filename'].isin(valid_files)]

# Save splits
df_train.to_csv('train_annotations_orthoptera.csv', index=False)
df_valid.to_csv('valid_annotations_orthoptera.csv', index=False)

# Print dataset stats
# Count labels in each dataset (using full filename, not prefix-based assignment)
num_train_1 = sum(df_train['is_orthoptera'] == 1)
num_train_0 = sum(df_train['is_orthoptera'] == 0)
num_valid_1 = sum(df_valid['is_orthoptera'] == 1)
num_valid_0 = sum(df_valid['is_orthoptera'] == 0)

print(f"Total files: {len(file_names)}")
print(f"Train: {len(train_files)} ({num_train_1} '1' labels, {num_train_0} '0' labels)")
print(f"Validation: {len(valid_files)} ({num_valid_1} '1' labels, {num_valid_0} '0' labels)")

#%% Doing a Random Forest model 
train_df_with_features = pd.read_csv('train_annotations_orthoptera.csv')
valid_df_with_features = pd.read_csv('valid_annotations_orthoptera.csv') 
X_train = train_df_with_features.drop(columns=["filename", "is_orthoptera"])
y_train = train_df_with_features["is_orthoptera"]
X_valid = valid_df_with_features.drop(columns=["filename", "is_orthoptera"])
y_valid = valid_df_with_features["is_orthoptera"]

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
with open('RFC_accuracy_Orthoptera.txt', 'w') as f:
    f.write(f"Validation Accuracy: {accuracy:.4f}\n")
    f.write(classification_report(y_valid, y_pred))
print("Classification report saved to RFC_accuracy_Orthoptera.txt")

