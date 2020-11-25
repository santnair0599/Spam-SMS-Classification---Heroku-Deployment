# Spam-SMS-Classification - Heroku-Deployment

## Table of Content
  * [Demo](#demo)
  * [Overview](#overview)
  * [Motivation](#motivation)
  * [Technical Aspect](#technical-aspect)
  * [Installation](#installation)
  * [Run](#run)
  * [Deployement on Heroku](#deployement-on-heroku)
  * [Directory Tree](#directory-tree)
  * [To Do](#to-do)
  * [Bug / Feature Request](#bug---feature-request)
  * [Technologies Used](#technologies-used)
  * [Team](#team)
  * [License](#license)
  * [Credits](#credits)
  
  ## Demo
Link: [https://sms-spam-classification-flask.herokuapp.com/](https://sms-spam-classification-flask.herokuapp.com/)

![Alt Text](https://github.com/santnair0599/Spam-SMS-Classification---Heroku-Deployment/blob/main/readme_resources/spam-sms-web-app.gif?raw=true)

## Overview
This is a simple text classification problem using python where we sort spam messages from ham. The dataset that we’re using can be found at https://www.kaggle.com/uciml/sms-spam-collection-dataset. The dataset contains one set of SMS messages in English of 5,574 messages, tagged as being ham (legitimate) or spam. The libraries used for building the model are — pandas, numpy, nltk, seaborn, matplotlib, string and sklearn. Finally, a python web API is created using Flask and the model is deployed in Heroku.

![alt text](https://github.com/santnair0599/Spam-SMS-Classification---Heroku-Deployment/blob/main/images/1.%20Dataset.png)

## Performance Metric
Since our dataset is imbalanced with 86% data being in ham class we’ll use F1 score (Precision*Recall/(Precision+Recall)) to measure the performance of the model. With the F1-score metric, we are trying to find an equal balance between precision and recall, which is extremely useful in most scenarios when we are working with imbalanced datasets. 

## Project Lifecycle
This project is divided into two part:

**1. Exploratory Data Analysis & Feature Engineering:** 
1. Created countplots for spam vs ham. It was observed that the dataset is imbalanced with 86% (4825 of 5525) of data belonging to ham class. The Spam and Ham data points were mapped with values as ham: 0 and spam: 1.

![alt text](https://github.com/santnair0599/Spam-SMS-Classification---Heroku-Deployment/blob/main/images/2.%20Imbalanced%20Dataset.png)

2. Oversampling method was used to handle the imbalanced dataset.

![alt text](https://github.com/santnair0599/Spam-SMS-Classification---Heroku-Deployment/blob/main/images/3.%20Balanced%20Dataset.png)

3. Created a new feature of total word count in each text message. Generated a plot showing distribution of word count for Ham messages versus distribution of word count of Spam messages. It was observed that the Ham messages word count range below 25 words whereas Spam messages word count range between 15 to 20 words. 

![alt text](https://github.com/santnair0599/Spam-SMS-Classification---Heroku-Deployment/blob/main/images/4.%20Word%20Count%20-%20Ham%26Spam.png)

4. Created a binary feature (0 or 1) representing whether a text message has currency symbol ('€', '$', '¥', '£', '₹'). Generated a countplot using this new feature. It was observed that about 1/3 of the Spam messages have currency symbols, whereas currency symbols are rarely found in Ham messages

![alt text](https://github.com/santnair0599/Spam-SMS-Classification---Heroku-Deployment/blob/main/images/6.%20Countplot%20-%20Currency%20Symbols.png)


**2. Data Cleaning:** The following activities were performed as part of data cleaning:
1. Removing special characters and numbers using regular expression
2. Converting the entire SMS into lower case
3. Tokenizing the SMS by words
4. Removing the stop words
5. Lemmatizing the words
6. Joining the lemmatized words
7. Building a corpus of words
8. Finally, created a bag of words representation using TfidfVectorizer

**3. Model Building and Evaluation:** 
Multinomial Naive bayes, Decision Tree Classifier, Random Forest Classifier, VotingClassifer (Using Decision Tree & Multinomial Naive Bayes algorithms and feeding it to voting algorithm to increase the F1 Score) models were build and the F1 scores are as shown below. 10-fold cross validation technique was used while building the models in order to flag problems like overfitting or selection bias and to give an insight on how the model will generalize to an independent dataset. It was noted that the Random Forest Classifier has the best F1 score, hence Random Forest Algorithm is selected for predicting the results of this problem statement.


 | Model  | F1 Score  |
| :------------: |:---------------:|
| Multinomial Naive Bayes     |  0.943 |
| Decision Tree Classifier       |  0.980        |
| **Random Forest Classifier** | **0.994**        |
| Voting Classifier | 0.980        |


![alt text](https://github.com/santnair0599/Spam-SMS-Classification---Heroku-Deployment/blob/main/images/7.%20Confusion%20Matrix%20-%20Random-Forest.png)


**Model Deployemnt:**   
1. Train a final RandomForestClassifier model, and generate pickle files for TfidfVectorizer and the trained RandomForestClassifier model.
2. Build a Flask web app named app.py and run the script.

![alt text](https://github.com/santnair0599/Spam-SMS-Classification---Heroku-Deployment/blob/main/images/9.%20app.py%20file.png)

3. Deploy the app at Heroku.

## Installation
The Code is written in Python 3.7. If you don't have Python installed you can find it [here](https://www.python.org/downloads/). If you are using a lower version of Python you can upgrade using the pip package, ensuring you have the latest version of pip.

**Virtualenv**

```
$ virtualenv venv
$ source ev/bin/activate
$ pip install -r requirements.txt
```

**Anaconda Enviroment**

```
$ conda create --name sms-spam-classifier python=3.7
$ conda activate sms-spam-classifier
$ pip install -r requirements.txt
```
or
```
$ conda env export > sms-spam-classifier.yml
$ conda env create -f sms-spam-classifier.yml
```
**Run**

```
$ export FLASK_APP=app.py
$ export FLASK_DEBUG=1
$ flask run
```
**Requirents.txt**

```
To install the required packages and libraries, run this command in the project directory after 
[cloning](https://www.howtogeek.com/451360/how-to-clone-a-github-repository/) the repository:

$ pip freeze > requirements.txt
```
or 
```
$ conda list -e > requirements.txt
```

**Deploying to heroku**
```
$ heroku create sms-spam-classifer --buildpack heroku/python
$ heroku git:remote -a sms-spam-classifer
$ pip install gunicorn
$ touch procfile
$ echo "web: gunicorn app:app --log-file=-" >> procfile
$ echo "python-3.7.1" >> runtime.txt
$ heroku local
$ heroku local web
$ git add .
$ git commit -m "commit msg"
$ git push heroku master
$ git push origin master
```

Also, refer the instructions given on [Heroku Documentation](https://devcenter.heroku.com/articles/getting-started-with-python) to deploy a web app.

```

```

## To Do
1. Build a Deep learning Spam Detection System for SMS using Keras and Python.
2. Use Twilio SMS APIs so that we will be able to classify SMS messages sent to the phone number you have registered in your Twilio account..

## Bug / Feature Request
If you find a bug (the website couldn't handle the query and / or gave undesired results), kindly open an issue [here](https://github.com/santnair0599/Spam-SMS-Classification---Heroku-Deployment/issues/new) by including your search query and the expected result.

If you'd like to request a new function, feel free to do so by opening an issue [here](https://github.com/santnair0599/Spam-SMS-Classification---Heroku-Deployment/issues/new). Please include sample queries and their corresponding results.


## Team
![Santosh Nair](https://avatars1.githubusercontent.com/u/31506535?s=400&amp;u=7940626a1196ca55b88b687d2aa84e043694b199&amp;v=4) |
-|
Santosh Nair


