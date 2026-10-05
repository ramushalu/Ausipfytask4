# TASK 4 - NETFLIX CONTENT SEGMENTATION
# Machine Learning Internship Project
#
# Objective:
# Group Netflix titles into meaningful content segments
# using K-Means Clustering.
#
# Workflow:
# 1. Prepare numerical and categorical features
# 2. Preprocess the data
# 3. Scale the features
# 4. Apply K-Means clustering
# 5. Identify content groups
# 6. Visualize clusters
# 7. Interpret cluster characteristics

# 1. IMPORT LIBRARIES

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from IPython.display import display

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler
)

from sklearn.impute import SimpleImputer

from sklearn.cluster import KMeans

from sklearn.decomposition import PCA

from sklearn.metrics import silhouette_score


print("=" * 70)
print("TASK 4 - NETFLIX CONTENT SEGMENTATION")
print("=" * 70)

# 2. LOAD DATASET

df = pd.read_csv("Dataset.csv")

print("\nDataset loaded successfully!")

print("Dataset shape:", df.shape)

# 3. BASIC DATA EXPLORATION

print("\n" + "=" * 70)
print("DATASET INFORMATION")
print("=" * 70)

print("\nFirst 5 rows:")

display(df.head())


print("\nColumn names:")

print(df.columns.tolist())


print("\nMissing values:")

print(df.isnull().sum())


print("\nDuplicate rows:")

print(df.duplicated().sum())

# 4. DATA CLEANING

df = df.drop_duplicates().copy()


# Fill categorical missing values

categorical_columns = [
    "type",
    "director",
    "country",
    "rating",
    "duration",
    "listed_in"
]


for column in categorical_columns:

    df[column] = df[column].fillna("Not Given")


# Convert release year to numeric

df["release_year"] = pd.to_numeric(
    df["release_year"],
    errors="coerce"
)


# Fill missing release year

df["release_year"] = df["release_year"].fillna(
    df["release_year"].median()
)


print("\nAfter cleaning:")

print("Dataset shape:", df.shape)

# 5. CREATE NUMERICAL FEATURES

# Extract numerical duration from values such as:
# "90 min" or "2 Seasons"

df["duration_number"] = (
    df["duration"]
    .astype(str)
    .str.extract(r"(\d+)")[0]
)

df["duration_number"] = pd.to_numeric(
    df["duration_number"],
    errors="coerce"
)


df["duration_number"] = df["duration_number"].fillna(
    df["duration_number"].median()
)


# Create number of genres

df["genre_count"] = (
    df["listed_in"]
    .astype(str)
    .apply(
        lambda x: len(
            [g for g in x.split(",") if g.strip()]
        )
    )
)


# Create number of countries

df["country_count"] = (
    df["country"]
    .astype(str)
    .apply(
        lambda x: len(
            [c for c in x.split(",") if c.strip()]
        )
    )
)


print("\nNew numerical features created:")

print("- duration_number")

print("- genre_count")

print("- country_count")

# 6. SELECT FEATURES

features = [
    "type",
    "release_year",
    "rating",
    "duration_number",
    "genre_count",
    "country_count"
]


X = df[features].copy()


print("\n" + "=" * 70)
print("FEATURES USED FOR CLUSTERING")
print("=" * 70)

print(features)

display(X.head())

# 7. IDENTIFY FEATURE TYPES

categorical_features = [
    "type",
    "rating"
]


numerical_features = [
    "release_year",
    "duration_number",
    "genre_count",
    "country_count"
]

# 8. PREPROCESS DATA

categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            )
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ]
)


numerical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        ),
        (
            "scaler",
            StandardScaler()
        )
    ]
)


preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            categorical_pipeline,
            categorical_features
        ),
        (
            "numerical",
            numerical_pipeline,
            numerical_features
        )
    ]
)

# 9. TRANSFORM DATA

X_processed = preprocessor.fit_transform(X)


print("\n" + "=" * 70)
print("DATA PREPROCESSING")
print("=" * 70)

print(
    "Original feature shape:",
    X.shape
)

print(
    "Processed feature shape:",
    X_processed.shape
)

# 10. FIND OPTIMAL NUMBER OF CLUSTERS

print("\n" + "=" * 70)
print("FINDING OPTIMAL NUMBER OF CLUSTERS")
print("=" * 70)


inertia_values = []

silhouette_values = []

cluster_range = range(2, 9)


for k in cluster_range:

    kmeans_temp = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    cluster_labels = kmeans_temp.fit_predict(
        X_processed
    )

    inertia_values.append(
        kmeans_temp.inertia_
    )

    silhouette_values.append(
        silhouette_score(
            X_processed,
            cluster_labels
        )
    )

# 11. ELBOW METHOD GRAPH

plt.figure(figsize=(10, 6))

plt.plot(
    list(cluster_range),
    inertia_values,
    marker="o"
)

plt.title(
    "Elbow Method for Selecting Number of Clusters"
)

plt.xlabel("Number of Clusters (K)")

plt.ylabel("Inertia")

plt.xticks(
    list(cluster_range)
)

plt.grid(True)

plt.tight_layout()

plt.show()

# 12. SILHOUETTE SCORE GRAPH

plt.figure(figsize=(10, 6))

plt.plot(
    list(cluster_range),
    silhouette_values,
    marker="o"
)

plt.title(
    "Silhouette Score for Different K Values"
)

plt.xlabel("Number of Clusters (K)")

plt.ylabel("Silhouette Score")

plt.xticks(
    list(cluster_range)
)

plt.grid(True)

plt.tight_layout()

plt.show()

# 13. DISPLAY K VALUES AND SCORES

cluster_evaluation = pd.DataFrame({

    "Number_of_Clusters": list(cluster_range),

    "Inertia": inertia_values,

    "Silhouette_Score": silhouette_values

})


print("\nCluster evaluation:")

display(cluster_evaluation)

# 14. SELECT K

# Select the K value with the highest silhouette score

best_k = int(
    cluster_evaluation.loc[
        cluster_evaluation[
            "Silhouette_Score"
        ].idxmax(),
        "Number_of_Clusters"
    ]
)


print(
    "\nBest K based on silhouette score:",
    best_k
)

# 15. TRAIN FINAL K-MEANS MODEL

print("\n" + "=" * 70)
print("TRAINING FINAL K-MEANS MODEL")
print("=" * 70)


kmeans_model = KMeans(
    n_clusters=best_k,
    random_state=42,
    n_init=10
)


cluster_labels = kmeans_model.fit_predict(
    X_processed
)


# Add cluster labels to original dataset

df["Cluster"] = cluster_labels


print(
    "\nK-Means clustering completed!"
)

# 16. CLUSTER DISTRIBUTION

print("\n" + "=" * 70)
print("CLUSTER DISTRIBUTION")
print("=" * 70)


cluster_counts = (
    df["Cluster"]
    .value_counts()
    .sort_index()
)


print(cluster_counts)

# 17. CLUSTER DISTRIBUTION GRAPH

plt.figure(figsize=(10, 6))

plt.bar(
    cluster_counts.index.astype(str),
    cluster_counts.values
)

plt.title(
    "Netflix Content Distribution by Cluster"
)

plt.xlabel("Cluster")

plt.ylabel("Number of Netflix Titles")

plt.tight_layout()

plt.show()

# 18. PCA FOR VISUALIZATION

print("\n" + "=" * 70)
print("PCA CLUSTER VISUALIZATION")
print("=" * 70)


# Convert sparse matrix to dense matrix if necessary

if hasattr(
    X_processed,
    "toarray"
):

    X_dense = X_processed.toarray()

else:

    X_dense = X_processed


pca = PCA(
    n_components=2,
    random_state=42
)


X_pca = pca.fit_transform(
    X_dense
)


print(
    "Explained variance by PCA components:"
)

print(
    pca.explained_variance_ratio_
)

# 19. PCA SCATTER PLOT

plt.figure(figsize=(12, 8))

scatter = plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=cluster_labels,
    alpha=0.6
)

plt.title(
    "Netflix Content Segmentation using K-Means"
)

plt.xlabel("Principal Component 1")

plt.ylabel("Principal Component 2")

plt.colorbar(
    scatter,
    label="Cluster"
)

plt.tight_layout()

plt.show()

# 20. CLUSTER CHARACTERISTICS

print("\n" + "=" * 70)
print("CLUSTER CHARACTERISTICS")
print("=" * 70)


cluster_summary = df.groupby(
    "Cluster"
).agg({

    "release_year": "mean",

    "duration_number": "mean",

    "genre_count": "mean",

    "country_count": "mean"

}).round(2)


cluster_summary["Number_of_Titles"] = (
    df["Cluster"]
    .value_counts()
    .sort_index()
)


display(cluster_summary)

# 21. MOST COMMON CONTENT TYPE BY CLUSTER

print("\n" + "=" * 70)
print("MOST COMMON CONTENT TYPE BY CLUSTER")
print("=" * 70)


cluster_type = pd.crosstab(
    df["Cluster"],
    df["type"]
)


display(cluster_type)


cluster_type_percentage = pd.crosstab(
    df["Cluster"],
    df["type"],
    normalize="index"
).round(3) * 100


print(
    "\nContent type percentage by cluster:"
)

display(cluster_type_percentage)

# 22. MOST COMMON RATING BY CLUSTER

print("\n" + "=" * 70)
print("MOST COMMON RATING BY CLUSTER")
print("=" * 70)


cluster_rating = pd.crosstab(
    df["Cluster"],
    df["rating"]
)


display(cluster_rating)

# 23. MOST COMMON GENRES BY CLUSTER

print("\n" + "=" * 70)
print("MOST COMMON GENRES BY CLUSTER")
print("=" * 70)


for cluster in sorted(
    df["Cluster"].unique()
):

    cluster_data = df[
        df["Cluster"] == cluster
    ]

    genre_series = (
        cluster_data["listed_in"]
        .dropna()
        .str.split(", ")
        .explode()
        .value_counts()
        .head(5)
    )

    print(
        f"\nCluster {cluster} - Top Genres:"
    )

    print(genre_series)

# 24. SILHOUETTE SCORE OF FINAL MODEL

final_silhouette = silhouette_score(
    X_processed,
    cluster_labels
)


print("\n" + "=" * 70)
print("FINAL CLUSTERING PERFORMANCE")
print("=" * 70)


print(
    "Selected number of clusters:",
    best_k
)

print(
    "Final Silhouette Score:",
    round(final_silhouette, 4)
)

print(
    "Final Inertia:",
    round(kmeans_model.inertia_, 2)
)

# 25. SHOW SAMPLE TITLES FROM EACH CLUSTER

print("\n" + "=" * 70)
print("SAMPLE TITLES FROM EACH CLUSTER")
print("=" * 70)


for cluster in sorted(
    df["Cluster"].unique()
):

    print(
        f"\nCluster {cluster}:"
    )

    sample_titles = df[
        df["Cluster"] == cluster
    ][
        [
            "title",
            "type",
            "rating",
            "release_year",
            "listed_in"
        ]
    ].head(5)

    display(sample_titles)

# 26. SAVE CLUSTERED DATASET

df.to_csv(
    "netflix_content_clusters.csv",
    index=False
)

# 27. SAVE CLUSTER SUMMARY

cluster_summary.to_csv(
    "netflix_cluster_summary.csv"
)

# 28. SAVE K-MEANS MODEL

joblib.dump(
    kmeans_model,
    "netflix_kmeans_model.pkl"
)


# Save preprocessing pipeline

joblib.dump(
    preprocessor,
    "netflix_clustering_preprocessor.pkl"
)


print("\n" + "=" * 70)
print("FILES SAVED SUCCESSFULLY")
print("=" * 70)

print("1. netflix_content_clusters.csv")
print("2. netflix_cluster_summary.csv")
print("3. netflix_kmeans_model.pkl")
print("4. netflix_clustering_preprocessor.pkl")

# 29. FINAL PROJECT SUMMARY

print("\n")
print("=" * 70)
print("TASK 4 - PROJECT SUMMARY")
print("=" * 70)

print(
    "Project: Netflix Content Segmentation"
)

print(
    "\nObjective:"
)

print(
    "Group Netflix titles into meaningful content "
    "segments using K-Means clustering."
)

print(
    "\nMachine Learning Technique:"
)

print(
    "K-Means Clustering"
)

print(
    "\nSelected Number of Clusters:",
    best_k
)

print(
    "\nFinal Silhouette Score:",
    round(final_silhouette, 4)
)

print(
    "\nTotal Netflix Titles:",
    len(df)
)

print(
    "\nFiles Generated:"
)

print(
    "- netflix_content_clusters.csv"
)

print(
    "- netflix_cluster_summary.csv"
)

print(
    "- netflix_kmeans_model.pkl"
)

print(
    "- netflix_clustering_preprocessor.pkl"
)

print("\n" + "=" * 70)

print(
    "TASK 4 COMPLETED SUCCESSFULLY!"
)

print("=" * 70)

Output:
======================================================================
TASK 4 - NETFLIX CONTENT SEGMENTATION
======================================================================

Dataset loaded successfully!
Dataset shape: (8790, 10)

======================================================================
DATASET INFORMATION
======================================================================

First 5 rows:
show_id	type	title	director	country	date_added	release_year	rating	duration	listed_in
0	s1	Movie	Dick Johnson Is Dead	Kirsten Johnson	United States	9/25/2021	2020	PG-13	90 min	Documentaries
1	s3	TV Show	Ganglands	Julien Leclercq	France	9/24/2021	2021	TV-MA	1 Season	Crime TV Shows, International TV Shows, TV Act...
2	s6	TV Show	Midnight Mass	Mike Flanagan	United States	9/24/2021	2021	TV-MA	1 Season	TV Dramas, TV Horror, TV Mysteries
3	s14	Movie	Confessions of an Invisible Girl	Bruno Garotti	Brazil	9/22/2021	2021	TV-PG	91 min	Children & Family Movies, Comedies
4	s8	Movie	Sankofa	Haile Gerima	United States	9/24/2021	1993	TV-MA	125 min	Dramas, Independent Movies, International Movies

Column names:
['show_id', 'type', 'title', 'director', 'country', 'date_added', 'release_year', 'rating', 'duration', 'listed_in']

Missing values:
show_id         0
type            0
title           0
director        0
country         0
date_added      0
release_year    0
rating          0
duration        0
listed_in       0
dtype: int64

Duplicate rows:
0

After cleaning:
Dataset shape: (8790, 10)

New numerical features created:
- duration_number
- genre_count
- country_count

======================================================================
FEATURES USED FOR CLUSTERING
======================================================================
['type', 'release_year', 'rating', 'duration_number', 'genre_count', 'country_count']
type	release_year	rating	duration_number	genre_count	country_count
0	Movie	2020	PG-13	90	1	1
1	TV Show	2021	TV-MA	1	3	1
2	TV Show	2021	TV-MA	1	3	1
3	Movie	2021	TV-PG	91	2	1
4	Movie	1993	TV-MA	125	3	1

======================================================================
DATA PREPROCESSING
======================================================================
Original feature shape: (8790, 6)
Processed feature shape: (8790, 20)

======================================================================
FINDING OPTIMAL NUMBER OF CLUSTERS
======================================================================



Cluster evaluation:
Number_of_Clusters	Inertia	Silhouette_Score
0	2	25873.691039	0.331079
1	3	20541.342261	0.353025
2	4	16108.036491	0.317934
3	5	13947.999449	0.310757
4	6	12456.503345	0.292171
5	7	11463.538777	0.292100
6	8	10855.113320	0.297030

Best K based on silhouette score: 3

======================================================================
TRAINING FINAL K-MEANS MODEL
======================================================================

K-Means clustering completed!

======================================================================
CLUSTER DISTRIBUTION
======================================================================
Cluster
0    2669
1    5511
2     610
Name: count, dtype: int64


======================================================================
PCA CLUSTER VISUALIZATION
======================================================================
Explained variance by PCA components:
[0.36396166 0.24153963]


======================================================================
CLUSTER CHARACTERISTICS
======================================================================
release_year	duration_number	genre_count	country_count	Number_of_Titles
Cluster					
0	2017.03	1.91	2.30	1.0	2669
1	2015.67	98.33	2.13	1.0	5511
2	1988.27	111.01	2.30	1.0	610

======================================================================
MOST COMMON CONTENT TYPE BY CLUSTER
======================================================================
type	Movie	TV Show
Cluster		
0	33	2636
1	5511	0
2	582	28

Content type percentage by cluster:
type	Movie	TV Show
Cluster		
0	1.2	98.8
1	100.0	0.0
2	95.4	4.6

======================================================================
MOST COMMON RATING BY CLUSTER
======================================================================
rating	G	NC-17	NR	PG	PG-13	R	TV-14	TV-G	TV-MA	TV-PG	TV-Y	TV-Y7	TV-Y7-FV	UR
Cluster														
0	1	0	4	1	1	2	728	96	1143	320	179	193	1	0
1	21	3	68	230	393	655	1277	117	1994	484	126	136	5	2
2	19	0	7	56	96	142	152	7	68	57	1	4	0	1

======================================================================
MOST COMMON GENRES BY CLUSTER
======================================================================

Cluster 0 - Top Genres:
listed_in
International TV Shows    1347
TV Dramas                  758
TV Comedies                565
Crime TV Shows             468
Kids' TV                   441
Name: count, dtype: int64

Cluster 1 - Top Genres:
listed_in
International Movies    2535
Dramas                  2152
Comedies                1472
Documentaries            827
Independent Movies       709
Name: count, dtype: int64

Cluster 2 - Top Genres:
listed_in
Dramas                  262
International Movies    205
Comedies                197
Action & Adventure      172
Classic Movies          115
Name: count, dtype: int64

======================================================================
FINAL CLUSTERING PERFORMANCE
======================================================================
Selected number of clusters: 3
Final Silhouette Score: 0.353
Final Inertia: 20541.34

======================================================================
SAMPLE TITLES FROM EACH CLUSTER
======================================================================

Cluster 0:
title	type	rating	release_year	listed_in
1	Ganglands	TV Show	TV-MA	2021	Crime TV Shows, International TV Shows, TV Act...
2	Midnight Mass	TV Show	TV-MA	2021	TV Dramas, TV Horror, TV Mysteries
5	The Great British Baking Show	TV Show	TV-14	2021	British TV Shows, Reality TV
17	Jailbirds New Orleans	TV Show	TV-MA	2021	Docuseries, Reality TV
18	Crime Stories: India Detectives	TV Show	TV-MA	2021	British TV Shows, Crime TV Shows, Docuseries

Cluster 1:
title	type	rating	release_year	listed_in
0	Dick Johnson Is Dead	Movie	PG-13	2020	Documentaries
3	Confessions of an Invisible Girl	Movie	TV-PG	2021	Children & Family Movies, Comedies
6	The Starling	Movie	PG-13	2021	Comedies, Dramas
7	Motu Patlu in the Game of Zones	Movie	TV-Y7	2019	Children & Family Movies, Comedies, Music & Mu...
8	Je Suis Karl	Movie	TV-MA	2021	Dramas, International Movies

Cluster 2:
title	type	rating	release_year	listed_in
4	Sankofa	Movie	TV-MA	1993	Dramas, Independent Movies, International Movies
29	Jeans	Movie	TV-14	1998	Comedies, International Movies, Romantic Movies
51	Minsara Kanavu	Movie	TV-PG	1997	Comedies, International Movies, Music & Musicals
53	Avvai Shanmughi	Movie	TV-PG	1996	Comedies, International Movies
60	Jaws	Movie	PG	1975	Action & Adventure, Classic Movies, Dramas

======================================================================
FILES SAVED SUCCESSFULLY
======================================================================
1. netflix_content_clusters.csv
2. netflix_cluster_summary.csv
3. netflix_kmeans_model.pkl
4. netflix_clustering_preprocessor.pkl


======================================================================
TASK 4 - PROJECT SUMMARY
======================================================================
Project: Netflix Content Segmentation

Objective:
Group Netflix titles into meaningful content segments using K-Means clustering.

Machine Learning Technique:
K-Means Clustering

Selected Number of Clusters: 3

Final Silhouette Score: 0.353

Total Netflix Titles: 8790

Files Generated:
- netflix_content_clusters.csv
- netflix_cluster_summary.csv
- netflix_kmeans_model.pkl
- netflix_clustering_preprocessor.pkl

======================================================================
TASK 4 COMPLETED SUCCESSFULLY!
======================================================================
