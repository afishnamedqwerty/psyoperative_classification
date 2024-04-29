



we are receiving reports that i am an institution


Cross-Origin Request Blocked: The Same Origin Policy disallows reading the remote resource at https://o922922.ingest.sentry.io/api/6537127/envelope/?sentry_key=9096873b2bd74da8bcc0495fc810a210&sentry_version=7&sentry_client=sentry.javascript.remix%2F7.110.1. (Reason: CORS request did not succeed). Status code: (null).





**The Paradigm-Shift of Social Spambots: Evidence, Theories, and Tools for the Arms Race**

Spambot Detection Methodologies

- Unsupervised Graph Clustering:
    - *Concept*: This approach involves constructing a graph where nodes represent Twitter accounts and edges reflect interactions or similarities (e.g., shared content, simultaneous actions).
    - *Technique*: The Markov Cluster Algorithm (MCL) or similar graph-based algorithms are used to detect densely connected subgraphs that likely represent coordinated spambot activities.
    - *Success Factors*: Graph clustering is effective because spambots, especially those designed to influence public opinion or amplify specific content, tend to operate in coordinated clusters, exhibiting more interconnectivity than normal users.

- Digital DNA Sequencing:
    - *Concept*: Each account's behavior is encoded as a sequence of symbolic representations (akin to DNA), where different types of tweets (e.g., retweets, mentions, content posts) are different 'genes'.
    - *Technique*: Similarities between these 'digital DNA' sequences are analyzed to detect groups of accounts with unusually high behavioral alignment, indicative of automated scripts or coordinated control.
    - *Success Factors*: This method hinges on the premise that while individual spambots may vary their behavior to evade detection, collectively, they still exhibit patterned behavior that can be captured through sequence alignment.

Speculated Novel Methodologies

**Behavioral Entropy Analysis**
    Technique
        - *Concept*: Measure the randomness in the account activities over time. Genuine users tend to have more variable behavior patterns compared to spambots, which might follow more predictable, scripted patterns.
        - *Application*: Implementing dynamic, adversarial models could theoretically stay one step ahead of spambot evolution, adjusting detection parameters in response to new spambot strategies.
        - *Entropy Calculation*: Entropy can be quantified using Shannon's formula 
    H(X)=−∑i=1n​P(xi​)logP(xi​), where P(xi)P(xi​) is the probability of event xixi​ (e.g., tweet types, interaction patterns).
        - *Time Series Analysis*: Analyze the time series data of account activities to measure the predictability and variability using entropy over sliding window periods, which helps identify automated scripting

    Existing Algos:
        - *Sequential Information Bottleneck (sIB): This algorithm can be used for clustering time-series data by compressing the data while preserving mutual information.
        - *Dynamic Bayesian Networks*: To model the probabilistic relationships between observed interactions over time, providing a framework to assess randomness and predictability in behaviors.

    Success Factors:
        - *Reduction of False Positives*: More accurately distinguishes between genuinely irregular human behavior and consistently patterned bot activities.
        - * Scalability and Adaptability*: Effective in dynamically changing social network environments due to its non-parametric nature.

**Adversarial ML**
    Technique
        - *Concept*: Utilize adversarial training techniques where detection systems continuously adapt against an evolving set of spambot tactics designed to circumvent them.
        - *Application*: Implementing dynamic, adversarial models could theoretically stay one step ahead of spambot evolution, adjusting detection parameters in response to new spambot strategies.
        - *Generative adversarial nickhurts (GANs)*: Train generative models to continually produce synthetic bot behaviors which are used to train the discriminative models, enhancing their detection capabilities.

    Existing Algos:
        - *Fast Gradient Sign Method (FGSM)*: Used for generating adversarial examples by utilizing the gradients of the neural network to create perturbations.
        - *DeepFool*: A more sophisticated method to efficiently compute perturbations that fool deep learning models.

    Success Factors:
        - *Continuous Improvement*: The system evolves as spambot strategies evolve, making it harder for spambots to find stable blind spots in detection mechanisms.
        - *Robustness Against Evasion*: Specifically addresses the challenge of spambots that are designed to circumvent typical detection methods.


**Network Traffic Analysis**
    Technique
        - *Concept*: Beyond analyzing content and metadata, examining the data flow patterns and IP address behaviors could reveal networks of spambots.
        - *Application*: Machine learning models could be trained to recognize suspicious network traffic patterns that are typical of centralized spambot operations but not of decentralized human activities.
        - *Anomaly Detection Models*: Use statistical models to identify anomalies in traffic patterns (e.g., bursts of activity from certain IP ranges) that correlate with spambot operations. maybe causal prediction by using the tweet DNA sequencing in tandem with IP range activity bursts
        - *Flow-Based Monitoring*: Implement monitoring at the IP flow level to observe both the volume and distribution of traffic, which can be indicative of centralized botnets.

    Existing Algos:
        - *Isolation Forest*: An algorithm effective for detecting anomalies in large datasets, particularly useful for high-dimensional network traffic data.
        - *Principle Component Analysis (PCA)*: Reduces the dimensionality of network traffic data, enhancing the visibility of anomalies.

    Success Factors:
        - *Early Detection*: Capable of identifying spambot networks before they can perform significant actions by monitoring at the network layer.
        - *High Accuracy & Low False Positives*: Effective in environments with diverse and voluminous traffic without overwhelming the system with false alarms.


**Cognitive Behavior Models**
    Technique
        - *Concept*: Incorporate models from cognitive science to predict natural human reactions and timing in social interactions, using deviations from these models to identify bots.
        - *Application*: Systems could be developed to flag accounts that consistently exhibit non-human-like reaction times or interaction patterns, which are hard for bots to accurately mimic.
        - *Cognitive Load Assessment*: Model the expected cognitive load in social interactions and flag accounts that deviate significantly from these expectations.
        - *Reaction Time Analysis*: Measure and analyze the timing patterns of account reactions to content, comparing them against human cognitive reaction time distributions.

    Existing Algos:
        - *Hidden Markov Models (HMM)*: To model the states of cognitive load based on observable actions.
        - *Time-Warped Distance Measures*: Utilized in analyzing the temporal sequences of reactions for alignment with human behavior patterns.

    Success Factors:
        - *Human-like Detection*: Specifically designed to identify bots trying to mimic human behavior, focusing on deeper interaction cues that are difficult to simulate accurately.
        - *Integration with Behavioral Cues*: Can be combined with textual and interaction analysis for a comprehensive profile of an account.

According to Spamhaus, the country with the most botnets is China, with over 590,000 bots. The US is second-worst, with 376,000 bots, followed closely by India, which has around 350,000.

Scalable and Generalizable Social Bot Detection through Data Selection

"Methods
considering limited behaviors, like retweeting and temporal
patterns, can only identify certain types of bots."

(2019) https://arxiv.org/pdf/1911.09179.pdf#cite.cresci2017paradigm

(Mar 2023) https://arxiv.org/abs/1701.03017

https://arxiv.org/pdf/1809.09684.pdf

https://www.comparitech.com/blog/information-security/ddos-statistics-facts/

Some autonomous system number (ASN) operators—mostly ISPs—also have larger numbers of infected IP addresses due to extensive botnet malware. A10 Networks writes that the top 5 ASNs with infected IP addresses are:

    China Telecom
    Charter Communications (US)
    Korea Telecom
    China Unicorn CN
    Chungwha Telecom  (China)





Reference Papers (potentially hallucinated i don't have portal access to these sites):

- Computational Propoganda in Iran: Analyzing Online Influence Operations
    - 2023
    - Journal of information Warfare
    - This paper provides a comprehensive analysis of the computational propaganda techniques used by Iran, focusing on the development and application of social media bots in spreading disinformation.

- Social Bots: The Dynamics of Political Influence in Iran
    - 2022
    - Political Communication
    - The authors discuss the dynamics of how social bots are utilized to manage public opinion in Iran and abroad, highlighting the technological and sociopolitical subtleties of these campaigns.

- The Role of Automated Accounts in Iranian Information Warfare
    - 2023
    - Cyber Conflict Studies Association
    - Link: 
    - This paper explores the strategic use of automated accounts in Iran's information warfare campaigns, providing a close examination of their operational tactics and strategic impacts.



    Embedding Techniques:
    Advanced machine learning models that utilize node and structural embeddings have become central in identifying bots within social networks. These methods involve creating representations of social media accounts that encapsulate their relational properties within the network, which can then be used to classify accounts as bots or not. This includes examining user metadata, tweet content, and the structure of their social network connections​ (SpringerOpen)​.

    Dataset Construction and Feature Utilization:
    The quality of data and the features used are crucial in training detection models. Datasets like TwiBot-20 incorporate a mix of semantic, property, and neighborhood information to represent accounts comprehensively. Featu res derived from these datasets include the number of followers, tweet frequency, and types of interactions, which are critical for accurate bot detection​ (SpringerOpen)​.

    Challenges with Bot Detection:
    Detection techniques face numerous challenges, including the adaptation of bots that mimic human behavioral patterns. Bots are becoming sophisticated enough to pass basic Turing tests, making them harder to detect based on content alone. Therefore, detection methods also focus on network behavior and metadata rather than just content analysis​ (First Monday)​.

    Utilization of AI and Machine Learning:
    Deep learning models, particularly those based on Transformer architectures, are at the forefront of distinguishing between bot-generated and human-generated content. These models analyze patterns of language use, engagement timing, and other subtleties that may indicate automated activity. The complexity of these models allows for an increasingly nuanced understanding of what constitutes bot-like activity on platforms like Twitter and Facebook​ (SpringerOpen)​​ (First Monday)​.

    Current Research and Development:
    Current research is not only focused on detection but also on understanding the intent behind bot activities—differentiating between benign bots (like chatbots for customer service) and malicious bots (used for misinformation). This involves a deeper analysis of the content's context, the account's network behavior, and changes in activity patterns over time​ (First Monday)​.



    Bot Detection Categories:
    - User profile metadata
    - Natural language features (NLP) from user tweets
    - Features extracted from the underlying social network

    TwiBot-20: twitter bot detection benchmark


    Scalable and Generalizable Social Bot Detection through Data Selection
    - Popular approaches leverage user, temporal, content, and social network features with random forest classifiers

    - Using supervised learning framework with random forest classifiers, attempting to achieve both the scalability of the faster methods and the accuracy of the feature-rich methods.
    - Unsupervised learning methods are used to find improbably similarities among accounts. No human labeled datasets are needed

    - Faster classiciation can be obtained using fewer features and logistic regression
    - tweet content including metadata to detect bots at the tweet level
    - a simpler approach is to test the randomness of the screen name



    Okay first we need to decide the appropriate learning framework for your project.
    - Supervised learning
        - Most supervised learning methods require manually labeled data
        - Popular approaches leverage user, temporal, content, and social network features with random forest classifiers




[Article] 1. Introduction

In this tutorial, we’ll study deterministic and stochastic optimization methods. We’ll focus on understanding the similarities and differences of these categories of optimization methods and describe scenarios where they are typically employed.

First, we’ll have a brief review of optimization methods. So, we’ll particularly explore the categories of deterministic and stochastic optimization methods, showing examples of algorithms for each. At last, we’ll compare both categories of optimization methods in a systematic summary.
2. Optimization in Computing

Computer Science is applicable to solve problems and improve processes in multiple areas of knowledge. We can do that by modeling problems and their inputs in a standard way, thus processing them with the proper problem-solving algorithm.

A typical use of problem-solving algorithms is to get optimized solutions for complex problems. As this brief explanation suggests, we call these algorithms of optimization algorithms.

In our context, optimization means the attempt to find an optimal solution to a problem.

However, a single problem usually has several potential solutions. In such a way, optimization algorithms evaluate objective functions to define which candidate solution is the best one.

The adopted objective function, in turn, changes according to the characteristics and constraints of the considered problem.

Moreover, it is not always guaranteed that an optimization algorithm finds the global optima result. Some optimization methods are exact. These always find global optima results. But, we also have non-exact optimization algorithms that may find the global optima result or only local optima ones.

At last, it is relevant to highlight that there is no globally best optimization algorithm. Choosing an optimization algorithm depends, among other aspects, on the problem characteristics, the admissible execution time and available computational resources, and the necessity of finding the global optima solution.

In the following sections, we’ll explore two categories of optimization algorithms: deterministic and stochastic.
3. Deterministic Optimization Algorithms

Deterministic optimization aims to find the global best result, providing theoretical guarantees that the returned result is the global best one indeed. To do that, deterministic optimization algorithms exploit particular and convenient features of a given problem.

Thus, deterministic optimization refers to complete or rigorous classes of algorithms in Neumaier’s classification:

    Complete: algorithms that reach global best results with an indefinitely long execution time. But, it is possible to evaluate and find a sufficient local best result in a finite time considering predefined tolerances
    Rigorous: algorithms that find global best results in a finite execution time and considering predefined tolerances

Deterministic optimization is particularly good for solving problems with clear exploitable features that help to compute the globally optimal result.

However, deterministic algorithms may have problems tackling black-box problems or extremely complex and volatile optimization functions. It occurs due to big searching spaces and the existence of intricate problem structures.
3.1. Examples of Deterministic Optimization Models

We have several classic algorithm models for implementing deterministic optimization algorithms. We briefly discuss some of them next.

The very first model of deterministic optimization is Linear Programming (LP). Linear programming consists of a mathematical model where a problem and its requirements are modeled through linear relationships and evaluated through linear objective functions.

On the other hand, we also have the Nonlinear Programming (NLP) model. In such a case, the problem, constraints, and objective functions may include nonlinear relationships. These problems are very challenging in the context of deterministic optimization.

Both LP and NLP are appropriate to solve convex problems (with a single optimal solution, which is the globally optimal one). But, we do also have options for non-convex optimization problems (with many candidates for optimal results).

Examples of other models of deterministic optimization are Integer Programming (IP), Non-convex Nonlinear Programming (NNLP), and Mixed-Integer Nonlinear Programming (MINLP).

The following image summarizes the deterministic optimization programming models according to the optimization problem features:
Deterministic 1

Examples of methods that implement deterministic optimization for these models are branch-and-bound, cutting plane, outer approximation, and interval analysis, among others.
4. Stochastic Optimization Algorithms

Stochastic optimization aims to reach proper solutions to multiple problems, similar to deterministic optimization. However, different from deterministic optimization, stochastic optimization algorithms employ processes with random factors to do so.

Due to these processes with random factors, stochastic optimization does not guarantee finding the optimal result for a given problem. But, there is always a probability of finding the globally optimal result.

Specifically, the probability of finding the globally optimal result in stochastic optimization relates to the available computing time. The probability of finding the globally optimal result increases as the execution time increases. So, if the execution time is infinite, we have a 100% chance of finding the global optima result.

In this way, it is impossible to have 100% certainty of finding the global optima result with stochastic algorithms in practice.

Despite this fact, there are many scenarios in real-life that a globally optimal result is not required. In such cases, simply reaching a result good enough with feasible time is sufficient. Thus, stochastic optimization can be naturally employed.

So, the most relevant advantage of stochastic optimization compared to deterministic optimization is the possibility of controlling the execution time: we can fastly find a result for a complex problem with a large search space (even if this result is a local optima one).
4.1. Heuristics and Metaheuristics

Several heuristics and metaheuristics use stochastic processes to find optimized results for a given problem.

In short, heuristics are strategies employed to solve a particular problem. Metaheuristics, in turn, are generic strategies adapted to solve multiple problems.

As heuristics depend on the problem, we do not have a generic example of stochastic heuristics algorithms. Metaheuristics, in turn, have many algorithms examples.

For metaheuristics, we can cite trajectory methods, such as tabu search, that may include stochastic decisions. Furthermore, we have population-based methods, such as genetic algorithms, grey-wolf optimization, and particle swarm optimization, that use several randomized processes.

The image next summarizes stochastic optimization according to its categories of algorithms:
Stochastic
5. Systematic Summary

Several areas of knowledge need to do some kind of optimization to solve particular problems. So, computing provides different optimization algorithms to cover this necessity.

Sometimes, we need to find the optimal result for a problem regardless of the time it takes. In such a case, deterministic optimization algorithms are the best choice.

But, we can have complex problems and a limited time to solve them. Thus, stochastic optimization algorithms are good choices since we can find proper solutions (even the globally optimal ones) in a feasible time.

Of course, deterministic and stochastic optimization include distinct characteristics and algorithms. The following table presents relevant features of deterministic and stochastic optimization.

Rendered by QuickLaTeX.com
6. Conclusion

In this article, we studied deterministic and stochastic optimization. First, we reviewed some general concepts regarding optimization in computing. After that, we specifically investigated deterministic and stochastic optimization through relevant characteristics, models, and algorithms. Finally, we compared deterministic and stochastic optimization in a systematic summary.

We can conclude that both deterministic and stochastic algorithms are crucial for solving problems computationally. If the globally optimal result is needed, we do require deterministic optimization. However, stochastic optimization is the best alternative if we only need to find a solution in a given time.