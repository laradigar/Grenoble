
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

embeds = pd.read_csv('/Users/laradiazgarcia/Desktop/minimalcode/Archive/embbedings_with_groups.csv')
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
folder_path = "/Users/laradiazgarcia/Desktop/minimalcode/Archive/All"
file_names = [f for f in os.listdir(folder_path) if f.endswith('.wav')]
df = pd.read_csv("/Users/laradiazgarcia/Desktop/minimalcode/Archive/is_orthoptera.csv")

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

# Define prefix-level split (instead of file-based split)
split_ratio = 0.7  # 70% train, 30% validation

num_train_1 = int(len(prefixes_1) * split_ratio)
num_train_0 = int(len(prefixes_0) * split_ratio)

train_prefixes_1 = set(prefixes_1[:num_train_1])
train_prefixes_0 = set(prefixes_0[:num_train_0])
train_prefixes = train_prefixes_1.union(train_prefixes_0)

valid_prefixes_1 = set(prefixes_1[num_train_1:])  # Remaining go to validation
valid_prefixes_0 = set(prefixes_0[num_train_0:])
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

#%% Extract features to train model using the Mel-frequency cepstrum
# This section calculates new features (MFCC) for the files instead of using our previous embeddings
train_df = pd.read_csv('train_annotations.csv')
valid_df = pd.read_csv('valid_annotations.csv')

def extract_features(file_path):
    """Extract MFCC features from an audio file."""
    try:
        y, sr = librosa.load(file_path, sr=None)  # Load audio file
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)  # Extract 13 MFCCs
        mfcc_mean = np.mean(mfcc, axis=1)  # Take the mean across time
        return mfcc_mean
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

# Function to extract features for the filenames in a given dataframe
def add_mfcc_features(df):
    mfcc_features = []

    for filename in df['filename']:
        file_path = os.path.join(folder_path, filename)
        mfcc_features.append(extract_features(file_path))

    # Add MFCC features as columns in the dataframe
    mfcc_df = pd.DataFrame(mfcc_features, columns=[f"mfcc_{i}" for i in range(13)])
    return pd.concat([df, mfcc_df], axis=1)

# Add MFCC features to the training and validation dataframes
train_df_with_features = add_mfcc_features(train_df)
valid_df_with_features = add_mfcc_features(valid_df)

# Save the updated dataframes to new CSV files
train_df_with_features.to_csv('train_with_mfcc.csv', index=False)
valid_df_with_features.to_csv('valid_with_mfcc.csv', index=False)

print("MFCC features added and saved to train_with_mfcc.csv and valid_with_mfcc.csv")


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
