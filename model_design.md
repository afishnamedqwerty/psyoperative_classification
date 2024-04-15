**Bot Detection Architecture: Multi-Layered Defense System (MLDS)**
***Path 1***

Integration and Action Mechanism
- *Cross-Layer Communication*: Ensure findings from one layer can inform analysis in another, creating a cohesive defense mechanism.
- *Bot classification and clustering*: Once bots are detected, automate classification and pre-processing necessary for clustering


Layer 1: Preprossing and Data Normalization
- *Objective*: Prepare and augment data for analysis
- *Techniques*: 
    - User Metadata Extraction: Gather comprehensive data on user accounts, including actvitiy patterns, network traffic data, and metadata
    - Content Analysis: Perform semantic analysis of the content share by accounts using Natural Language Processing (NLP) to identify spam-like patterns
    - Traffic Analysis: Collect network-level data to spot anomalous traffic patterns indicative of botnet coordination.

Layer 2: Individual Account Analysis
- *Objective*: Analyze individual account behaviors to flag potential bots.
- *Techniques*: 
    - Behavioral Entropy Analysis: Use Shannon's entropy to quantify predictability in account activities. 
    - Adversarial Machine Learning (AML): Continuously refine detection models using adversarial examples to catch bots attempting to evade detection.
    - Cognitive Behavioral Models: Assess accounts for deviations from human cognitive and reaction time patterns.

Layer 3: Group Behavior Analysis
- *Objective*: Identify coordinated bot actvities by analyzing group behaviors.
- *Techniques*: 
    - Digital DNA Sequencing: Encode and compare the behavioral patterns of user groups, identifying bots through unnatural sequence similarities. 
    - Graph Clustering: Use graph-based algorithms like Markov Cluster Algorithm (MCL) to identify densely connected groups indicative of botnets.
    - Anomaly Detection in Collective Actions: Detect anomalies in collective behavviors (like synchronized posting) that signal botnet activity.

Layer 4: Network Traffic and Anomaly Detection
- *Objective*: Leverage network traffic analysis to spot botnets
- *Techniques*: 
    - Flow-Based Monitoring: Monitor IP flow data to catch bursts of activity or other anomalies associated with botnets.
    - PCA for Traffic Analysis: Apply Principle Component Analysis to reduce dimensionality and highlight network traffic anomalies.

Layer 5: Continuous Learning and Adaptation
- *Objective*: Ensure the system evolvves to keep pace with new botnet strategies.
- *Techniques*: 
    - Reinforcement Learning (RL): Implement RL to adapt detection strategies based on feedback loops from detected bot activities.
    - Crowdsourcing Feedback: Incorporate user reports and feedback into the learning model to improve detection accuracy.




***Path 2***

Layer 1: Preprossing and Data Normalization
- *Objective*: Prepare and standardize subsequent layers
- *Techniques*: 
    - Data Collection: Aggregate user metadata, content data (tweets, retweets, mentions), and interaction data (likes, replies).
    - Normalization: Standardize data formats and normalize timestamps to a unified timezone.
    - Noise Reduction: Filter out irrelevant data elements to focus on features significant for analysis. 

Layer 2: Individual Account Analysis
- *Objective*: Analyze individual account behaviors to flag potential bots.
- *Techniques*: 
    - Feature Extraction: Extract features such as tweet frequency, content diversity, account creation date, follower-to-following ratio, and metadata completeness.
    - Anomaly Detection (Using Isolation Forest): Identify accounts whose features significantly deviate from typical human user distributions.
    - Supervised Learning (using SVM or Random Forest): Classify accounts based on labeled data from known human and bot accounts 

Layer 3: Behavioral Entropy Analysis:
- *Objective*: Assess the predictability of account behaviors over time.
- *Techniques*: 
    - Entropy Measurement: Calculate the Shannon entropy for sequences of account actions to detect low variability in behavior indicative of automated scripts.
    - Temporal Analysis: Use time series analysis to examine the consistency and timing of posts and interactions, identifying unnatural patterns.
    - Supervised Learning (using SVM or Random Forest): Classify accounts based on labeled data from known human and bot accounts 

Layer 4: Group Behavior Analysis
- *Objective*: Detect bots based on collective behavior of account clusters
- *Techniques*: 
    - Graph Construction: Build interaction graphs based on retweets, mentions, and shared links.
    - Community Detection (using Graph Clustering): Apply Principle Component Analysis to reduce dimensionality and highlight network traffic anomalies.
    - Digital DNA Sequencing: Analyze groups for suspiciously similar digital DNA sequences, highlighting potential bot clusters.

Layer 5: Adversarial Feedback Loop
- *Objective*: Continuously improve detection algorithms through adversarial learning.
- *Techniques*: 
    - Adversarial Training: Regularly generate synthetic bot behavior data (using GANs) to test and refine detection models.
    - Feedback System: Integrate feedback from new detections to update model parameters and adapt to evolving botnet strategies.

Layer 6: Decision & Action
- *Objective*: Final determination and response to detected bots
- *Techniques*: 
    - Voting System: Employ a majority voting system across the outputs of different models to make final bot determination.
    - Action Protocol: Automated reporting and blocking of bots, with alerts sent to human operators for high-risk cases.
    - Digital DNA Sequencing: Analyze groups for suspiciously similar digital DNA sequences, highlighting potential bot clusters.



**Custom Dynamic Graph CNN Architecture (DGCNN-BotDetect)**

*Architecture Overview*

The DGCNN-BotDetect model will be structured to dynamically update both its graph structure and node embeddings to reflect the changing nature of Twitter interactions, focusing on heterogeneous subgraphs that include users, tweets, hashtags, and URLs. The architecture will consist of the following components:

1. Data Preprocessing and Dynamic Graph Construction

    - Real-time Data Streaming: 
        - Implement a data ingestion module that continuously collects Twitter data, including tweets, retweets, replies, mentions, and follows.

    - Graph Construction: 
        - Nodes represent users, tweets, hashtags, and URLs.
        - Edges represent interactions such as mentions, retweets, and replies, dynamically added or removed as interactions occur.

    - Heterogeneous Subgraph Sampling: 
        - Continuously sample subgraphs around nodes of interest (potential bots) to capture extended neighborhoods and their evolution.

2. Feature Engineering

    - Node Feature Extraction:
        - Extract textual features using NLP techniques like TF-IDF or word embeddings.
        - Metadata features including user account properties (creation date, verified status, etc.), activity patterns (tweet frequency, common posting times), and engagement metrics (like-to-tweet ratios, average retweets).
    - Edge Feature Extraction:
        - Type of interaction (mention, retweet, reply), temporal features (time of interaction), and interaction frequency.
    - Temporal Feature Encoding: Encode time-series data to capture temporal dynamics of interactions, using techniques like Fourier transforms or recurrent neural networks.

3. Dynamic Graph Convolutional Layers

    - Edge Learning and Updating:
        - Trainable edge weights that evolve based on the learning feedback, using attention mechanisms to adjust the importance of different types of interactions dynamically.
    - Temporal Graph Convolution
        - Apply convolutional operations that not only aggregate spatial neighbor features but also integrate temporal dynamics.
        - Use a combination of spatial-temporal convolutional networks to process these features.

4. Heterogeneous Attention Mechanisms

    - Node-Type Specific Attention: 
        - Different attention heads for different types of nodes, allowing the model to learn unique embeddings for users, tweets, hashtags, and URLs based on their roles in the graph.
    - Temporal Attention: 
        - Focus on changes in interaction patterns over time, which are critical for detecting bots activated for specific campaigns or events.

5. Output and Decision Layers

    - Readout Layer: 
        - Aggregate node embeddings from various graph convolutional layers to form a graph-level representation, which will be used for the final bot detection.
    - Classification Layer:
        - A fully connected layer followed by a softmax to classify subgraphs as either bot-driven or legitimate.

6. Model Training and Evaluation

    - Loss Functions:
        - Binary cross-entropy loss for classification
        - Additional regularization terms to manage the complexity and overfitting of the graph model

    - Model Evaluation:
        - Regular evaluation intervals using validation data that includes known bot and legitimate account interactions.
        - Metrics: Accuracy, Precision, Recall, F1-score, and ROC AUC.

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


    TwiBot-22 (requires data request): https://github.com/LuoUndergradXJTU/TwiBot-22

    Cresci-17: https://botometer.osome.iu.edu/bot-repository/datasets.html

    pronbot2 (link in bio disinfectant): https://github.com/r0zetta/pronbot2

    MGTAB: https://github.com/GraphDetec/MGTAB/tree/main

    MIB datasets (requires data request): http://mib.projects.iit.cnr.it/dataset.html
    