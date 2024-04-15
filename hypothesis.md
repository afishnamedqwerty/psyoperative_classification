Hypothesis Formulation
Start by formulating a hypothesis regarding the presence of bot-generated content in LLM training
data and its potential effects on bias within the models' output. For example, 'Bot-generated content
in social media and forums, when included in the training data of LLMs, introduces or amplifies
specific biases within the model's output.'

Outline types of bot-generated content relevant to LLM training data

evidence of social media scraping for LLM training

evidence of LLM (or any model) showing abnormal behavior following training on poisoned data - preferably proof of feature accuracy degradation


Audiobot side tangent I-Iuawei's ahead of us by a week: https://arxiv.org/pdf/2404.04904.pdf

Preprocess training datasets ensuring graph neural network compatibility
-  Accuracy, F1-Score and Precision are used as evaluation metrics and experiments are conducted on three benchmark datasets.




Bot Testing Data collection options
- User network collection: Use breadth-first search (BFS) for user collection starting from "seed users." We augment a two-stage data sampling process per TwiBot-22 paper
    - Distribution diversity: Given user metadata such as follower count, different types of users fall differently into the metadata distribution. Distribution diversity aims to sample users in the top, middle, and bottom of the distribution. For numerical metadata, among a user’s neighbors and their metadata values, we select the k users with the highest value, k with the lowest, and k randomly chosen from the rest. For true-or-false metadata, we select k with true and k with false.
    - Value diversity: Given a user and its metadata, value diversity is adopted so that neighbors with significantly different metadata values are more likely to be included, ensuring the diversity of collected users. For numerical metadata, among expanding user u’s neighbors X and their metadata values xnum, the probability of user x ∈ X being sampled is denoted as p(x) ∝ |unum − xnum|. For true-or-false metadata we select k users from the opposite class.