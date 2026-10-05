# Netflix Content Segmentation

## Project Overview

This project focuses on segmenting Netflix content into different groups using **K-Means Clustering**.

The goal is to identify groups of Netflix content based on similarities in the available content features.

## Objectives

* Prepare the Netflix dataset for clustering.
* Select relevant features for segmentation.
* Preprocess and scale the data.
* Apply the K-Means clustering algorithm.
* Analyze the generated clusters.
* Save the clustering model and results.

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn
* Joblib

## Methodology

### 1. Data Preparation

The Netflix dataset is loaded and prepared for the clustering process.

### 2. Feature Selection

Relevant features from the Netflix dataset are selected for segmentation.

### 3. Data Preprocessing

The selected features are preprocessed and scaled so that different feature ranges do not affect the clustering process.

### 4. K-Means Clustering

The **K-Means clustering algorithm** is applied to group similar Netflix content into different clusters.

### 5. Cluster Analysis

The generated clusters are analyzed to understand the characteristics and distribution of Netflix content in each group.

## Generated Files

The following files are generated:

* `netflix_content_clusters.csv`
* `netflix_cluster_summary.csv`
* `netflix_kmeans_model.pkl`
* `netflix_clustering_preprocessor.pkl`

## Results

The K-Means algorithm successfully segments Netflix content into groups based on the selected features. The cluster results help identify similarities and patterns within the Netflix content dataset.

## Conclusion

This project demonstrates the use of **unsupervised machine learning** for Netflix content segmentation using the K-Means clustering algorithm.
