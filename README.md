# Fake-News-Detection
In this project, I have used different NLP techniques and machine learning algorithms to classify fake news articles using sci-kit libraries from python.

first we will also need to download and install below 3 packages after the installation of either python or anaconda from these URL:
[can refer to this url https://www.python.org/downloads/ to download python.
To install anaconda check this url https://www.anaconda.com/download/]

Then need to have these 3 packages-->
1-Sklearn (scikit-learn)
2-numpy
3-scipy

 
-->run below commands in command prompt/terminal to install these packages for python:
pip install -U scikit-learn
pip install numpy
pip install scipy

-->for anaconda then run below commands in anaconda prompt to install these packages:
conda install -c scikit-learn
conda install -c anaconda numpy
conda install -c anaconda scipy

**Dataset used**
The data source used for this project is LIAR dataset which contains 3 files with .tsv format for test, train and validation. 
Below is some description about the data files used for this project.

**LIAR: A BENCHMARK DATASET FOR FAKE NEWS DETECTION**

the original dataset contained 13 variables/columns for train, test and validation sets as follows:

Column 1: the ID of the statement ([ID].json).
Column 2: the label. (Label class contains: True, Mostly-true, Half-true, Barely-true, FALSE, Pants-fire)
Column 3: the statement.
Column 4: the subject(s).
Column 5: the speaker.
Column 6: the speaker's job title.
Column 7: the state info.
Column 8: the party affiliation.
Column 9-13: the total credit history count, including the current statement.
9: barely true counts.
10: false counts.
11: half true counts.
12: mostly true counts.
13: pants on fire counts.
Column 14: the context (venue / location of the speech or statement).

To make things simple I have chosen only 2 variables from this original dataset for this classification. The other variables can be added later to add some more complexity and enhance the features.

Below are the columns used to create 3 datasets that have been in used in this project

Column 1: Statement (News headline or text).
Column 2: Label (Label class contains: True, False)
You will see that newly created dataset has only 2 classes as compared to 6 from original classes. Below is method used for reducing the number of classes.

Original -- New
True -- True
Mostly-true -- True
Half-true -- True
Barely-true -- False
False -- False
Pants-fire -- False
The dataset used for this project were in csv format named train.csv, test.csv and valid.csv and can be found in repo. The original datasets are in "liar" folder in tsv format.

**File descriptions**

**DataPrep.py**
This file contains all the pre processing functions needed to process all input documents and texts. First I read the train, test and validation data files then performed some pre processing like tokenizing, stemming etc. There are some exploratory data analysis is performed like response variable distribution and data quality checks like null or missing values etc.

**FeatureSelection.py**
In this file I have performed feature extraction and selection methods from sci-kit learn python libraries. For feature selection, I have used methods like simple bag-of-words and n-grams and then term frequency like tf-tdf weighting. I have also used word2vec and POS tagging to extract the features, though POS tagging and word2vec has not been used at this point in the project.

**classifier.py**
Here I have build all the classifiers for predicting the fake news detection. The extracted features are fed into different classifiers. I have used Naive-bayes, Logistic Regression, Linear SVM, Stochastic gradient descent and Random forest classifiers from sklearn. Each of the extracted features were used in all of the classifiers. Once fitting the model, I compared the f1 score and checked the confusion matrix. After fitting all the classifiers, 2 best performing models were selected as candidate models for fake news classification. I have performed parameter tuning by implementing GridSearchCV methods on these candidate models and chosen best performing parameters for these classifier. Finally selected model was used for fake news detection with the probability of truth.

**prediction.py**
My finally selected and best performing classifier was Logistic Regression which was then saved on disk with name final_model.sav. Once close this repository, this model will be copied to user's machine and will be used by prediction.py file to classify the fake news. It takes an news article as input from user then model is used for final classification output that is shown to user along with probability of truth.
