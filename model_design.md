As i see it three major parts:

- Initial generative-bot activity classification model
    - Use the prepared datasets above to build a classifier using posting frequency, sentence complexity, and use of generice phrases among other statistically relevant features to establish a base model for proper binary identification of generative-bot content. 

- Broad randomized Twitter subsets for model clustering and evaluation
    - Focusing on twitter as any social media dataset we want to perform classification must contain our initial features. I'm still investigated unsupervised learning for now and think supervised is the best route. I say this (very much still reading into it lmao) because X paper adopts supervised learning with random forest classifiers, "attempting to achieve both the scalability of the faster methods and the accuracy of the feature-rich methods."

    *Scalable and Generalizable Social Bot Detection through Data Selection* https://arxiv.org/pdf/1911.09179.pdf

    They do this by collecting the embedded user object from each tweet. Two advantages with this: no extra queries needed for detection and user objects contain archived user profile information. Meaning if someone tweets and then changes their username you can still see the original username in the tweet's rmbedded user object. This paper is from 2019 i need to verify user object still exists with this years API v2 changes. They also cover screen_names as a useful feature for bot detection - stating "the likelihood of a screen name is defined by the geometric-mean likelihood of all bigrams in it... ...Tests show the likelihood feature can effectively distinguish random strings from authentic screen names."

- A pipeline for parsing and clustering our real-world (alernative?) data from our Twitter subsets
    - Cluster the user/tweet activity metadata to graph bot-network activity per subset
    - Yes this is barebones the categorization is bothering me im reading more into this part of the pipeline


- A pipeline for parsing and clustering our real-world (alernative?) data from our Twitter subsets
    - Cluster the user/tweet activity metadata to graph bot-network activity per subset
    - Yes this is barebones the categorization is bothering me im reading more into this part of the pipeline


K-Means Clustering

    Purpose: To partition users into preliminary groups based on clearly quantifiable and linearly separable features like tweet frequency, follower counts, and timestamps.
    Advantages: Fast and efficient for large datasets, providing a good baseline segmentation.
    Process:
        Scale features using StandardScaler or MinMaxScaler.
        Determine the optimal number of clusters using the Elbow Method or Silhouette Score.
        Run K-Means to classify data points into clusters.

Hierarchical Clustering

    Purpose: To build upon K-Means results by identifying nested clusters, allowing us to observe the data at different levels of granularity.
    Advantages: Does not require the number of clusters to be specified a priori, useful for detailed exploratory analysis.
    Process:
        Use the output of K-Means as input features to reduce dimensionality.
        Apply agglomerative clustering with linkage methods like Ward, which minimizes the variance within clusters.

DBSCAN (Density-Based Spatial Clustering of Applications with Noise)

    Purpose: To identify dense clusters of bot-like activities and isolate anomalies or outliers such as sophisticated bots that do not fit into common activity patterns.
    Advantages: Capable of finding arbitrarily shaped clusters and handling noise.
    Process:
        Apply DBSCAN following hierarchical clustering to refine clusters and detect outliers.
        Choose eps and min_samples based on nearest-neighbor distances to optimize cluster formation.

HDBSCAN (Hierarchical DBSCAN)

    Purpose: An extension of DBSCAN that works better for data of varying densities, which is typical in social media datasets where some bots are highly active while others are not.
    Advantages: Automatically infers the best clustering parameters and offers stability over different data samples.
    Process:
        Employ HDBSCAN to further enhance the clustering quality from DBSCAN, particularly useful for handling data with varying density.

Community Detection Algorithms (Graph-Based)

    Purpose: To explore interaction networks (followers, retweets, mentions) and detect communities within these networks, which can highlight coordinated bot activities.
    Advantages: Provides insights into complex relational structures that are not easily observable with traditional clustering.
    Process:
        Convert the Twitter interactions into a graph format.
        Apply algorithms like the Louvain method to identify communities based on modularity optimization.

Visualization and Interpretation

    Network Visualization: Use tools like Gephi or NetworkX to visualize the clusters formed by graph-based methods.
    Dimensionality Reduction for Visualization: Employ PCA or t-SNE to reduce the feature space for visualizing the results of high-dimensional clusters.