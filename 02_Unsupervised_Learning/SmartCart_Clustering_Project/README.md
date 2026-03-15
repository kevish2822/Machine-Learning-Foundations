# SmartCart Customer Segmentation — Machine Learning Study

## Overview

This project focuses on customer segmentation using unsupervised machine learning
techniques. The goal is to group customers based on their purchasing behavior,
demographics, and spending patterns in order to identify meaningful customer
segments.

Customer segmentation helps businesses better understand their customers and
design targeted marketing strategies.

The project applies clustering algorithms to identify distinct customer groups
based on engineered behavioral features.

---

## Dataset Description

Dataset used: **SmartCart Customer Dataset**

The dataset contains information about customers including:

- Demographic information
- Income level
- Customer tenure
- Spending across different product categories
- Household composition
- Marketing response behavior

The goal is to identify groups of customers with similar characteristics and
purchasing patterns.

---

## Data Preprocessing

Several preprocessing steps were performed to prepare the dataset for clustering.

These include:

- Handling missing values
- Feature engineering
- Removing unnecessary columns
- Detecting and handling outliers
- Encoding categorical variables
- Feature scaling

---

## Feature Engineering

New features were created to improve clustering performance:

- **Age** derived from birth year
- **Total Children** combining kid and teen household counts
- **Total Spending** aggregating spending across multiple product categories
- **Customer Tenure** derived from customer joining date

These engineered features better represent customer behavior.

---

## Dimensionality Reduction

Principal Component Analysis (PCA) was applied to reduce dimensionality and
visualize the data in lower-dimensional space.

PCA helps reveal patterns and structures in the dataset that may not be easily
visible in higher dimensions.

---

## Determining the Optimal Number of Clusters

Two techniques were used to determine the best number of clusters:

### Elbow Method

The Elbow Method was used to analyze the within-cluster sum of squares (WCSS)
and identify the point where adding more clusters provides diminishing returns.

### Silhouette Score

Silhouette Score was used to measure how well data points fit within their
assigned clusters compared to other clusters.

---

## Clustering Algorithms Implemented

Two clustering algorithms were applied:

### K-Means Clustering

- Partitions data into K clusters
- Minimizes distance between points and cluster centroids
- Works well for spherical clusters

### Agglomerative Clustering

- Hierarchical clustering method
- Builds clusters by iteratively merging similar groups
- Does not require centroid initialization

---

## Cluster Characterization

After clustering, customer segments were analyzed by calculating average
feature values for each cluster.

This helps interpret cluster behavior and identify distinct customer profiles
such as:

- High-income high-spending customers
- Low-income moderate-spending customers
- Younger customers with lower spending patterns

These insights can help businesses tailor marketing strategies and improve
customer engagement.

---

## Key Learning Outcomes

- Understanding unsupervised learning techniques
- Applying clustering algorithms to real-world datasets
- Feature engineering for behavioral data
- Determining optimal cluster numbers
- Using PCA for dimensionality reduction
- Interpreting clusters to derive business insights

---

## Project Files

- `SmartCart_Project.ipynb` → Complete analysis and clustering workflow

---

## Dependencies

Required libraries include:

- Python
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn
- Kneed