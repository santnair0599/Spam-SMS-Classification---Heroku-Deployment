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
This is a simple text classification problem using python where we sort spam messages from ham. The dataset that we’re using can be found at https://www.kaggle.com/uciml/sms-spam-collection-dataset. The dataset contains one set of SMS messages in English of 5,574 messages, tagged as being ham (legitimate) or spam. The modules used for building the model are — pandas, numpy, nltk, seaborn, matplotlib, string and sklearn. Finally, a python web API is created using Flask and the model is deployed in Heroku.

## Performance Metric
Since our dataset is imbalanced with 86% data being in ham class we’ll use F1 score (Precision*Recall/(Precision+Recall)) to measure the performance of the model. With the F1-score metric, we are trying to find an equal balance between precision and recall, which is extremely useful in most scenarios when we are working with imbalanced datasets. 

## Project Lifecycle
This project is divided into two part:

**1. Exploratory Data Analysis & Feature Engineering:** 
1. Created countplots for spam vs ham. It was observed that the dataset is imbalanced with 86% (4825 of 5525) of data belonging to ham class. The Spam and Ham data points were mapped with values as ham: 0 and spam: 1. Oversampling method was used to handle the imbalanced dataset.
2. Created a new feature of total word count in each text message. Generated a plot showing distribution of word count for Ham messages versus distribution of word count of Spam messages. It was observed that the Ham messages word count range below 25 words whereas Spam messages word count range between 15 to 20 words. 
3. Created a binary feature (0 or 1) representing whether a text message has currency symbol ('€', '$', '¥', '£', '₹'). Generated a countplot using this new feature. It was observed that about 1/3 of the Spam messages have currency symbols, whereas currency symbols are rarely found in Ham messages

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

**1. Model Creation and Hyperparameter Tuning:** 
**1. Model Deployemnt:**   




Training a deep learning model using Keras. (_Not covered in this repo. I'll update the link here once I make it public._)
2. Building and hosting a Flask web app on Heroku.
    - A user can choose image from a device or capture it using a pre-built camera.
    - Used __Amazon S3 Bucket__ to store the uploaded image and predictions.
    - Used __CSRF Token__ to protect against CSRF attacks.
    - Used __Sentry__ to catch the exception on the back-end.
    - After uploading the image, the predictions are displayed on a __Bar Chart__.

## Installation
The Code is written in Python 3.7. If you don't have Python installed you can find it [here](https://www.python.org/downloads/). If you are using a lower version of Python you can upgrade using the pip package, ensuring you have the latest version of pip. To install the required packages and libraries, run this command in the project directory after [cloning](https://www.howtogeek.com/451360/how-to-clone-a-github-repository/) the repository:
```bash
pip install -r requirements.txt
```

## Run
> STEP 1
#### Linux and macOS User
Open `.bashrc` or `.zshrc` file and add the following credentials:
```bash
export AWS_ACCESS_KEY="your_aws_access_key"
export AWS_SECRET_KEY="your_aws_secret_key"
export ICP_BUCKET='your_aws_bucket_name'
export ICP_BUCKET_REGION='bucket_region'
export ICP_UPLOAD_DIR='bucket_path_to_save_images'
export ICP_PRED_DIR='bucket_path_to_save_predictions'
export ICP_FLASK_SECRET_KEY='anything_random_but_unique'
export SENTRY_INIT='URL_given_by_sentry'
```
Note: __SENTRY_INIT__ is optional, only if you want to catch exceptions in the app, else comment/remove the dependencies and code associated with sentry in `app/main.py`

#### Windows User
Since, I don't have a system with Windows OS, here I collected some helpful resource on adding User Environment Variables in Windows.

__Attention__: Please perform the steps given in these tutorials at your own risk. Please don't mess up with the System Variables. It can potentially damage your PC. __You should know what you're doing__. 
- https://www.tenforums.com/tutorials/121855-edit-user-system-environment-variables-windows.html
- https://www.onmsft.com/how-to/how-to-set-an-environment-variable-in-windows-10

> STEP 2

To run the app in a local machine, shoot this command in the project directory:
```bash
gunicorn wsgi:app
```

## Deployement on Heroku
Set the environment variable on Heroku as mentioned in _STEP 1_ in the __Run__ section. [[Reference](https://devcenter.heroku.com/articles/config-vars)]

![](https://i.imgur.com/TmSNhYG.png)

Our next step would be to follow the instruction given on [Heroku Documentation](https://devcenter.heroku.com/articles/getting-started-with-python) to deploy a web app.

## Directory Tree 
```
├── app 
│   ├── __init__.py
│   ├── main.py
│   ├── model
│   ├── static
│   └── templates
├── config
│   ├── __init__.py
├── processing
│   ├── __init__.py
├── requirements.txt
├── runtime.txt
├── LICENSE
├── Procfile
├── README.md
└── wsgi.py
```

## To Do
1. Convert the app to run without any internet connection, i.e. __PWA__.
2. Add a better vizualization chart to display the predictions.

## Bug / Feature Request
If you find a bug (the website couldn't handle the query and / or gave undesired results), kindly open an issue [here](https://github.com/rowhitswami/Indian-Currency-Prediction/issues/new) by including your search query and the expected result.

If you'd like to request a new function, feel free to do so by opening an issue [here](https://github.com/rowhitswami/Indian-Currency-Prediction/issues/new). Please include sample queries and their corresponding results.

## Technologies Used

![](https://forthebadge.com/images/badges/made-with-python.svg)

[<img target="_blank" src="https://keras.io/img/logo.png" width=200>](https://keras.io/) [<img target="_blank" src="https://flask.palletsprojects.com/en/1.1.x/_images/flask-logo.png" width=170>](https://flask.palletsprojects.com/en/1.1.x/) [<img target="_blank" src="https://number1.co.za/wp-content/uploads/2017/10/gunicorn_logo-300x85.png" width=280>](https://gunicorn.org) [<img target="_blank" src="https://www.kindpng.com/picc/b/301/3012484.png" width=200>](https://aws.amazon.com/s3/) 

[<img target="_blank" src="https://sentry-brand.storage.googleapis.com/sentry-logo-black.png" width=270>](https://www.sentry.io/) [<img target="_blank" src="https://openjsf.org/wp-content/uploads/sites/84/2019/10/jquery-logo-vertical_large_square.png" width=100>](https://jquery.com/)

## Team
[![Santosh Nair](https://avatars1.githubusercontent.com/u/31506535?s=400&amp;u=7940626a1196ca55b88b687d2aa84e043694b199&amp;v=4)](https://rohitswami.com/) |
-|
[Santosh Nair](https://rohitswami.com/) |)

## License
[![Apache license](https://img.shields.io/badge/license-apache-blue?style=for-the-badge&logo=appveyor)](http://www.apache.org/licenses/LICENSE-2.0e)

Copyright 2020 Rohit Swami

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

## Credits
- [Google Images Download](https://github.com/hardikvasa/google-images-download) - This project wouldn't have been possible without this tool. It saved my enormous amount of time while collecting the data. A huge shout-out to its creator [Hardik Vasa](https://github.com/hardikvasa).
