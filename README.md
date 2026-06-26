Project Overview

This repository is a data science and machine learning project focused on classifying ecoacoustics sounds for the project “Study of the relationship between acoustic and functional diversity”, funded by the Royal Society grant International Exchanges 2024. This work was done as a visiting researcher at the Laboratoire d'Écologie Alpine (CNRS) (Grenoble,
France. January 2025 – February 2025)

Project Structure & Key Files

The repository is organised into several key directories, each representing a different phase or approach to the classification problem.

UMAPs/: Code for creating UMAP (Uniform Manifold Approximation and Projection) plots. UMAP is a dimensionality reduction technique often used in exploratory data analysis to visualise high-dimensional data (like spectrograms of sound events) in 2D or 3D.

binary_LGBM/: This directory contains the work for a Light Gradient Boosting Machine (LGBM) classifier. LGBM is a fast, distributed, high-performance gradient boosting framework that is well-suited for binary classification tasks. 

binary_RFC/: This directory contains the work for a Random Forest Classifier (RFC). This is another robust machine learning algorithm used for classification. 

mantis/: This directory contains some of the algorithms from other folders applied to Mantis data.

Models and Progress

Baseline Model: A Random Forest Classifier (binary_RFC) was implemented as a strong, interpretable baseline model.

Advanced Model: A Gradient Boosting Machine (binary_LGBM) was then developed  to evaluate whether it could improve performance and faster training times compared to the RFC.

Languages and Tools
Python 100%: The entire project is written in Python, the most common language for machine learning and audio analysis.

Key Libraries:

umap-learn for the UMAP visualisations.

lightgbm for the LGBM model.

scikit-learn for the Random Forest model and other utilities.

librosa and pydub for audio processing and feature extraction.

Project Status
Active Development: Hoping to use it on more data from the LECA databases.

Getting Started
To run this project locally, you would need to:

Clone the repository.

Install the required Python dependencies.

Navigate to the relevant directory (e.g., binary_LGBM/) and run the Python scripts.
