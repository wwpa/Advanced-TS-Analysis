# Advanced Sentiment Analysis and Text Classification Framework in Metaverse of Project.

## Introduction
Welcome to our project repository! This project focuses on sentiment analysis and text classification using multiple well-known datasets. The datasets used in this project include IMDB Reviews, Reuters-21578, SMS Spam Collection, 20 Newsgroups, and Shakespear's Novels. This README provides an overview of each dataset, including a brief description and links to the sources.

## Project Overview
This project focuses on performing sentiment analysis and text classification using several well-known datasets, integrating machine learning and deep learning algorithms to achieve high accuracy results. As part of the ongoing project, some of the researched content has been included in the paper. The core logic of the algorithms, implementation details, and concepts are as follows. Due to the specialized nature of the project, some specifics are not extensively covered. 

### Algorithm and Techniques:  
-[BETR_CNM_Pseudo.pdf](https://github.com/user-attachments/files/16112305/BETR_CNM_Pseudo.pdf)
-![image](https://github.com/wwpa/Advanced-TS-Analysis/assets/174091478/66659cd1-64c0-47d5-821f-8d04420f89bb)
- Natural Language Processing (NLP): Utilizes NLP techniques for preprocessing and feature extraction of text data.
- Bayesian Machine Learning Models: Provides probabilistic modeling for sentiment analysis and text classification.
- Deep Learning Techniques: Combines RNN and CNN to effectively analyze the emotional flow and topic distribution of text data.
- Machine Learning
- Bayesian Networks
- CNN (Convolutional Neural Network)
- RNN (Recurrent Neural Network)
- LSTM (Long Short-Term Memory)

### Key Approaches
This algorithm is designed to improve the accuracy of topic and sentiment analysis as well as text classification, and includes the following key approaches:

[Mathematical Formulations]
1. Bayes' theorem :
$P(T, E \mid D)=\frac{P(D \mid T, E) P(T) P(E)}{P(D)}$
2. Conditional probability using RNN outputs:
$P(E \mid S ; \theta)=\frac{\exp \left(\Psi\left(s, e, h_t ; \Theta\right)\right)}{\sum_{e^{\prime}} \exp \left(\Psi\left(s, e^{\prime}, h_t ; \Theta\right)\right)}$
3. Conditional probability for topics using CNN outputs:
$P(T \mid D ; \theta)=\frac{\exp \left(\operatorname{CNN}\left(D ; \theta_T\right)\right)}{\sum_{T^{\prime}} \exp \left(\operatorname{CNN}\left(D ; \theta_{T^{\prime}}\right)\right)}$
4. Model relevance of feature vector vD to topic T:
$P(D \mid T)=\sigma\left(W_T \cdot v_D+b_T\right)$
5. Posterior probability for topics and emotions:
$P(T \mid D, S ; \Theta)=\frac{\left(\exp \left(\phi\left(T, v_D, H_s ; \Theta\right)\right)\right)}{\left(\sum_{T^{\prime}}\left[\exp \left(\phi\left(\hat{T}, v_D, H_s ; \Theta\right)\right)\right]\right)}$

### BETR-CNM Model
The BETR-CNM model (Bayesian Emotional Topic Recurrent-Convolutional Neural Network) integrates CNN, RNN, and Bayesian networks to analyze topics and sentiments in text data. 
It starts with initializing an embedding layer and transforming words into vectors. 
CNNs extract feature vectors from documents, while RNNs model sentence sentiments using LSTM layers. 
The model constructs a Bayesian network to compute the joint probability of topics and emotions, integrating CNN and RNN outputs to calculate prior and posterior probabilities. Model optimization involves tuning CNN, RNN, and Bayesian parameters through cross-validation. Sentiment analysis differentiates emotions in sentences, and topic analysis identifies patterns to estimate topic associations. The model outputs comprehensive topic and emotion distributions for decision-making and advanced multi-text analysis.

## Datasets
We evaluated the performance of sentiment analysis and text classification across various domains using the IMDB review dataset, Reuters-21578 dataset, SMS Spam Collection, and 20 Newsgroups dataset.

### 1. IMDB Reviews Dataset
Description: The IMDB Reviews dataset, also known as the Large Movie Review Dataset, contains 50,000 movie reviews. These reviews are split into 25,000 for training and 25,000 for testing. Each review is labeled as positive or negative, making this dataset suitable for binary sentiment classification tasks. The dataset is available in both raw text format and a preprocessed bag-of-words format.
Source: This dataset was introduced by Andrew L. Maas et al. (2011) in their paper on learning word vectors for sentiment analysis.
•	TensorFlow IMDB Reviews(https://www.tensorflow.org/datasets/catalog/imdb_reviews)
________________________________________________________________________________
### 2. Reuters-21578 Dataset
Description: The Reuters-21578 dataset is a collection of documents that appeared on the Reuters newswire in 1987. It contains 21,578 documents that are categorized into multiple classes, making it suitable for text classification tasks across various domains.
Source: This dataset is available from the UCI Machine Learning Repository.
•	UCI Machine Learning Repository - Reuters-21578([https://archive.ics.uci.edu/ml/datasets/reuters-21578+text+categorization+collection](https://archive.ics.uci.edu/dataset/137/reuters+21578+text+categorization+collection))
________________________________________________________________________________
### 3. SMS Spam Collection Dataset
Description: The SMS Spam Collection dataset is a public set of SMS labeled messages that have been collected for mobile phone spam research. It includes both spam and non-spam messages, making it suitable for binary classification tasks.
Source: This dataset is available from the UCI Machine Learning Repository.
•	UCI Machine Learning Repository - SMS Spam Collection(https://archive.ics.uci.edu/ml/datasets/sms+spam+collection)
________________________________________________________________________________
### 4. 20 Newsgroups Dataset
Description: The 20 Newsgroups dataset is a collection of approximately 20,000 newsgroup documents, partitioned evenly across 20 different newsgroups. It is commonly used for text classification and clustering tasks.
Source: This dataset can be accessed through scikit-learn or Jason Rennie's page.
•	Scikit-Learn 20 Newsgroups(https://scikit-learn.org/stable/datasets/real_world.html#newsgroups-dataset)
•	Jason Rennie's Page(http://qwone.com/~jason/20Newsgroups/)
________________________________________________________________________________
### 5. Classical English Literature
Description: Text from the notable literature works.
Source: Various online repositories provide access to them, for example, [Shakespeare's Works] etc.
[shak_1.json](https://github.com/user-attachments/files/16112628/shak_1.json)
[shak_2.json](https://github.com/user-attachments/files/16112634/shak_2.json)
[shak_3.json](https://github.com/user-attachments/files/16112630/shak_3.json)
[whit.json](https://github.com/user-attachments/files/16112631/whit.json)

## How to Cite
If you use these datasets in your research, please cite the original authors and sources as mentioned in the dataset descriptions above.
- Shakespeare's Works: (https://www.opensourceshakespeare.org/)
- [Whitman's Leaves of Grass]: The Walt Whitman Archive. Available at: [Whitman's Leaves of Grass](https://www.whitmanarchive.org/published/LG/index.html)

## Visualization
![figure 4](https://github.com/wwpa/Advanced-TS-Analysis/assets/174091478/5b2baace-bf62-4bc3-89a0-cc234c1a4f38)
![figure 10](https://github.com/wwpa/Advanced-TS-Analysis/assets/174091478/e0119bd9-ecc8-4114-ae79-41d8812cd0f7)
![figure 9](https://github.com/wwpa/Advanced-TS-Analysis/assets/174091478/adad03f1-3572-45d9-a28c-5ac7cdd48fb3)
![figure 8](https://github.com/wwpa/Advanced-TS-Analysis/assets/174091478/b5ce9927-12dd-47bf-b4e6-fa5b8bc4924e)
![figure 7](https://github.com/wwpa/Advanced-TS-Analysis/assets/174091478/0a23cf24-5aa8-4efb-a7a1-2db3d962cf89)
![figure 6](https://github.com/wwpa/Advanced-TS-Analysis/assets/174091478/18cde484-20f8-4c1e-83ea-b7e516ced0dd)
![figure 5](https://github.com/wwpa/Advanced-TS-Analysis/assets/174091478/fa744067-c302-4b68-b143-03ed08d7963d)

## PDF Version
Due to confidentiality, please refer to the research paper for detailed information on the algorithm and methodology.










