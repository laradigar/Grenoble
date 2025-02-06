"""
This script performs UMAP embedding on a dataset and visualizes the results using scatter plots.

The script performs the following steps:
1. Reads embeddings data from a CSV file.
2. Applies UMAP to reduce the dimensionality of the data.
3. Creates a DataFrame with the UMAP results and corresponding group labels.
4. Explodes the group labels to handle multiple labels per data point.
5. Plots the UMAP embeddings with different group labels.
6. Filters and plots specific groups of interest (Orthoptera, Diptera, Hemiptera, Hymenoptera, Insecta).
7. Filters and plots specific groups of interest (Diptera, Hymenoptera).
8. Reads another CSV file with different category labels.
9. Applies UMAP to the new data and plots the results with category labels.

Functions:
- None

Dependencies:
- numpy
- pandas
- matplotlib
- seaborn
- umap

Usage:
- Ensure the required CSV files ('embbedings_with_groups.csv', 'embeds_groups_cats.csv') are in the same directory as the script.
- Run the script to generate and save the UMAP embedding plots.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import umap
#groups = ["Anthropophony", "Aves", "Geophony", "Mammalia", "Orthoptera", "Hemiptera", "Diptera", "Hymenoptera", "Anura", "Insecta"]

embeds = pd.read_csv('embbedings_with_groups.csv') 
data = embeds.iloc[:,2:514]
fit = umap.UMAP()
u = fit.fit_transform(data)
u_df = pd.DataFrame(u)
u_df["group"] = embeds.group
u_df["group"] = u_df["group"].str.split(', ')
u_df = u_df.explode("group", ignore_index=True)
u_df.columns = ['X', 'Y', 'group']

plt.figure()
sns.scatterplot(data=u_df, x='X', y='Y', hue='group', palette='Set2', alpha=0.5, s=1)
#sns.stripplot(data=u_df, x='X', y='Y', hue='group', jitter=True, palette='Set2', alpha=0.5, s=1)
plt.title('UMAP embedding')
plt.legend(title='Group', bbox_to_anchor=(1.05, 1), loc='upper left', markerscale=5)
plt.tight_layout()
plt.axis('off')
plt.savefig("all.png", dpi=1000, bbox_inches='tight')
plt.show()

## This is for comparing only the data from Orthoptera, Diptera, Hemiptera, Hymenoptera and Insecta
## Basically a closeup of the existing embeddings

bugs = ["Orthoptera", "Diptera", "Hemiptera", "Hymenoptera", "Insecta"]
bugs_df = u_df[u_df['group'].isin(bugs)]
shubugs_df = bugs_df.sample(frac=1, random_state=42).reset_index(drop=True)

plt.figure()
sns.scatterplot(data=shubugs_df, x='X', y='Y', hue='group', palette='Set2', alpha=0.5, s=2)
#sns.stripplot(data=u_df, x='X', y='Y', hue='group', jitter=True, palette='Set2', alpha=0.5)
plt.title('UMAP embedding: Arthropods')
plt.legend(title='Group', bbox_to_anchor=(1.05, 1), loc='upper left', markerscale=5)
plt.tight_layout()
plt.axis('off')
plt.savefig("arthropods_OG.png", dpi=1000, bbox_inches='tight')
plt.show()

## Now only Diptera and Hymenoptera
bugs = ["Diptera", "Hymenoptera"]
bugs_df = u_df[u_df['group'].isin(bugs)]
shubugs_df = bugs_df.sample(frac=1, random_state=42).reset_index(drop=True)

plt.figure()
sns.scatterplot(data=shubugs_df, x='X', y='Y', hue='group', palette='Set2', alpha=0.5, s=2)
plt.title('UMAP embedding: Diptera vs. Hymenoptera')
plt.legend(title='Group', bbox_to_anchor=(1.05, 1), loc='upper left', scatterpoints=1, markerscale=5)
plt.tight_layout()
plt.axis('off')
plt.savefig("diptera_hymenoptera.png", dpi=1000, bbox_inches='tight')
plt.show()

## Colouring the same embeddings but now looking a focal/soundscape
embeds = pd.read_csv('embeds_groups_cats.csv')
data = embeds.iloc[:,1:513]
fit = umap.UMAP()
u = fit.fit_transform(data)
u_df = pd.DataFrame(u)
u_df["category"] = embeds.category
u_df = u_df.set_axis(['X', 'Y', 'category'], axis=1)
shu_df = u_df.sample(frac=1, random_state=42).reset_index(drop=True)

plt.figure()
sns.scatterplot(data=shu_df, x='X', y='Y', hue='category', palette='Set2', alpha=0.5, s=5)
plt.title('UMAP embedding')
plt.legend(title='Category', bbox_to_anchor=(1.05, 1), loc='upper left', markerscale=5)
plt.tight_layout()
plt.axis('off')
plt.savefig("high_quality_plot.png", dpi=1000, bbox_inches='tight')
plt.show()