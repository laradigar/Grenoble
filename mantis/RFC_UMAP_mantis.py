#%% 
# Import modules and libraries
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
from sklearn.metrics import accuracy_score, classification_report, f1_score, recall_score
import matplotlib.pyplot as plt
import joblib
import random
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import umap

#%% THIS IS SINGING INSECTS
# Doing a Random Forest model for binary classification of singing insects
train_df_with_features = pd.read_csv('train_annotations_singing.csv')
valid_df_with_features = pd.read_csv('valid_annotations_singing.csv') 

# Check that there is no overlap in the file names between datasets
def extract_prefix(filename):
    return "_".join(filename.split("_")[:-3])  # Remove everything after the last 2 underscores

# Create sets of prefixes
train_prefixes = set(train_df_with_features["filename"].apply(extract_prefix))
valid_prefixes = set(valid_df_with_features["filename"].apply(extract_prefix))

# Drop filename column and split into X and y
X_train = train_df_with_features.drop(columns=["filename", "is_singing"])
y_train = train_df_with_features["is_singing"]
X_valid = valid_df_with_features.drop(columns=["filename", "is_singing"])
y_valid = valid_df_with_features["is_singing"]

#%% THIS IS AVES
train_df_with_features = pd.read_csv('/Users/laradiazgarcia/Desktop/Grenoble/binary_RFC/train_annotations_aves.csv')
valid_df_with_features = pd.read_csv('/Users/laradiazgarcia/Desktop/Grenoble/binary_RFC/valid_annotations_aves.csv') 

# Create sets of prefixes
train_prefixes = set(train_df_with_features["filename"].apply(extract_prefix))
valid_prefixes = set(valid_df_with_features["filename"].apply(extract_prefix))

X_train = train_df_with_features.drop(columns=["filename", "is_aves"])
y_train = train_df_with_features["is_aves"]
X_valid = valid_df_with_features.drop(columns=["filename", "is_aves"])
y_valid = valid_df_with_features["is_aves"]

#%% This is the actual RFC
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)  # 100 trees
rf_model.fit(X_train, y_train)

# Make predictions on the validation set
threshold = 0.5
y_pred = rf_model.predict_proba(X_valid)
y_pred = (y_pred[:, 1] >= threshold).astype(int)

final_eval_df = pd.DataFrame({"y_true": y_valid, "y_pred": y_pred})
# Evaluate model performance
# Counting the number of false positives to see if there's any even though precision is 1.00
accuracy = accuracy_score(y_valid, y_pred)
print(f"Validation Accuracy: {accuracy:.4f}")
# Print a full classification report
print(classification_report(y_valid, y_pred))

#%% Predictions on new ORCHAMP embeddings
data_ORC = pd.read_csv('/Users/laradiazgarcia/Desktop/Grenoble/mantis/mantis_home_orchamp_sound_2022_ARM_1520_1.csv', sep=";")
X_ORC = data_ORC.iloc[:,2:514]

# Make predictions on the ORCHAMP dataset
threshold = 0.4
y_ORC = rf_model.predict(X_ORC)

# Get probability estimates for the test set
y_probs = rf_model.predict_proba(X_ORC)
y_probs_1 = (y_probs[:,1] >= threshold).astype(int)

# Display the probability estimates
for i, prob in enumerate(y_probs):
    print(f"Sample {i+1}: Class 0 Probability = {prob[0]:.4f}, Class 1 Probability = {prob[1]:.4f}")

#%% UMAP
fit = umap.UMAP()
u = fit.fit_transform(X_ORC)
u_df = pd.DataFrame(u)
u_df["is_aves"] = y_probs_1
u_df["filename"] = data_ORC["filename"]
u_df.columns = ['X', 'Y', 'is_aves', "filename"]
plt.figure()
sns.scatterplot(data=u_df, x='X', y='Y', hue="is_aves", palette=['#0f1f99', '#35bc11'], alpha=0.7, s=1)
plt.title('UMAP embedding')
plt.tight_layout()
plt.axis('off')
plt.legend(title='Is it a bird?', loc='upper right', markerscale=5)
plt.savefig("mantis_umap_is_aves.png", dpi=1000, bbox_inches='tight')
plt.show()

#%% UMAP for time of day
# Extract the hhmmss part from the filename
data_ORC['time'] = data_ORC['filename'].str.extract(r'_(\d{6})\.wav$')[0]
u_df['time'] = data_ORC['time'].astype(int)
# Apply the condition for daytime (060000 to 210000) but it is UTC and should be two hours later!
u_df['is_daytime'] = ((u_df['time'] >= 40000) & (u_df['time'] <= 190000)).astype(int)
# Drop the intermediate 'time' column if not needed
u_df.drop(columns=['time'], inplace=True)
u_df.columns = ['X', 'Y', 'is_aves', 'filename','is_daytime']

plt.figure()
sns.scatterplot(data=u_df, x='X', y='Y', hue="is_daytime", palette=['#0f1f99', '#ffa33c'], alpha=0.7, s=1)
plt.title('UMAP embedding')
plt.tight_layout()
plt.axis('off')
plt.legend(title='Is it daytime?', loc='upper right', markerscale=5)
plt.savefig("mantis_umap_is_daytime.png", dpi=1000, bbox_inches='tight')
plt.show()

#%% UMAP for day of recording
# Extract the hhmmss part from the filename
u_df['date'] = u_df['filename'].str.split('_').str[1]

plt.figure()
sns.scatterplot(data=u_df, x='X', y='Y', hue="date", alpha=0.7, s=1)
plt.title('UMAP embedding')
plt.tight_layout()
plt.axis('off')
plt.legend(title='What day is it?', loc='upper right', markerscale=5)
plt.savefig("mantis_umap_date.png", dpi=1000, bbox_inches='tight')
plt.show()


#%% What are my outliers? Copy the files from the ORCHAMP_SON
df = u_df.loc[(u_df['X'] >= 5.5)]
df.to_csv('outliers.csv', index=False)

import shutil
import csv
# Define paths
csv_file = "outliers_filenames.csv"  # Update with your CSV file path
source_folder = "/VOLUMES/ORCHAMP_SON/ORCHAMP_SON_2022/recs/ARM1"  # Update with the folder containing the files
destination_folder = "/Users/laradiazgarcia/Desktop/Grenoble/mantis/outliers"  # Update with the target folder

# Ensure the destination folder exists
os.makedirs(destination_folder, exist_ok=True)

# Read the CSV file and copy files
with open(csv_file, newline='', encoding='utf-8') as file:
    reader = csv.reader(file)
    for row in reader:
        if row:  # Ensure it's not an empty row
            full_path = row[0].strip()
            filename = os.path.basename(full_path)  # Extract only the filename

            source_path = os.path.join(source_folder, filename)
            destination_path = os.path.join(destination_folder, filename)

            if os.path.exists(source_path):
                shutil.copy2(source_path, destination_path)  # Copy with metadata
                print(f"Copied: {filename}")
            else:
                print(f"File not found: {filename}")