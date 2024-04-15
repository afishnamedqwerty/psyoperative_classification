# Bot Detection Architecture for Twibot-22 Dataset

## Overview

This project presents a comprehensive bot detection system designed to accurately identify and classify bot-driven activities within the Twibot-22 dataset on Twitter. It leverages advanced machine learning techniques and graph theory principles to differentiate between human and bot accounts based on behavior, content, and network structure.

## Architecture Design

The architecture is divided into several key components:

### Data Preprocessing
- Data Loading: Load users, tweets, lists, and hashtags data from the Twibot-22 dataset.
- Data Cleaning: Address missing values, outliers, and normalize datetime formats.
- Feature Extraction: Derive meaningful attributes from user metadata and content interactions.

### Graph Construction
- Create a directed graph representing the diverse relationships between different entities (users, tweets, etc.).
- Utilize networkx for graph operations and storage.

### Dynamic Subgraph Sampling
- Implement random walks to sample relevant subgraphs dynamically.
- Focus on local graph structures to capture behavioral patterns indicative of bots.

### Feature Augmentation
- Behavioral: Tweet frequencies, retweet ratios, and other temporal behaviors.
- Content-based: Use of hashtags, URLs, and content sensitivity.
- Account Characteristics: Verification status, account age, and profile completeness.
- Network Attributes: Followers-to-following ratio and interactions with verified accounts.

### Model Architecture
- Enhanced Graph Convolutional Network (GCN) with Graph Attention Network (GAT) layers.
- Global mean pooling to derive graph-level features.
- Fully connected layers for classification.

### Training and Evaluation Framework
- Loss Function: Cross-entropy for classification tasks.
- Optimizer: Adam for stochastic optimization.
- Evaluation Metrics: Accuracy, Precision, Recall, F1-score, ROC-AUC.

## Main Components and Methodologies

1. **Graph Neural Networks (GNN)**: Employing GCN and GAT to process graph-structured data and learn from the network topology.
2. **Natural Language Processing (NLP)**: Utilizing BERT embeddings to analyze tweet content and derive semantic features.
3. **Incremental Learning**: Updating model weights as new data arrives without the need for retraining from scratch.
4. **Adversarial Training**: Including adversarial examples during training to improve model robustness against sophisticated bots.

## Metrics and Visualization

### Key Metrics for Testing
- **Accuracy**: The ratio of correctly predicted observations to the total observations.
- **Precision**: The ratio of correctly predicted positive observations to the total predicted positives.
- **Recall (Sensitivity)**: The ratio of correctly predicted positive observations to the all observations in actual class.
- **F1 Score**: The weighted average of Precision and Recall.
- **ROC-AUC**: The Area Under the Receiver Operating Characteristic Curve from prediction scores.

### Visualization Tools
- **Confusion Matrix**: Visualizing the performance of the algorithm with the actual and predicted classifications.
- **ROC Curve**: Displaying the diagnostic ability of the binary classifier system as its discrimination threshold is varied.
- **Feature Importance**: Bar chart showing the contribution of each feature to the model prediction.

## Conclusion

This bot detection system provides a robust and dynamic approach to identifying bot activity within the Twitter landscape. It is designed to be scalable and adaptable, accommodating the ever-evolving nature of social bots.

