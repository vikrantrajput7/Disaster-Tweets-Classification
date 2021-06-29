# Disaster Tweets Real or Not Classification

## Introduction

Twitter has become an important communication channel in times of emergency.
The ubiquitousness of smartphones enables people to announce an emergency they’re observing in real-time. Because of this, more agencies are interested in programatically monitoring Twitter (i.e. disaster relief organizations and news agencies).<br>
But, it’s not always clear whether a person’s words are actually announcing a disaster. Take this example:<br>
<img src="https://user-images.githubusercontent.com/26309477/123736954-bd3a4d80-d8bf-11eb-869b-5d94eb534838.png" width="200" /> <br>
The author explicitly uses the word “ABLAZE” but means it metaphorically. This is clear to a human right away, especially with the visual aid. But it’s less clear to a machine.

In this task, we are challenged to build a machine learning model that predicts which Tweets are about real disasters and which one’s aren’t.

This is an ongoing competition on Kaggle.  
<cite> https://www.kaggle.com/c/nlp-getting-started </cite>

## Dataset
### 1. Description
<p>
<b>id </b>- a unique identifier for each tweet
  
<b>text </b>- the text of the tweet
  
<b>location </b>- the location the tweet was sent from (may be blank)
  
<b>keyword </b>- a particular keyword from the tweet (may be blank)
  
<b>target </b>- in train.csv only, this denotes whether a tweet is about a real disaster (1) or not (0)
</p>

Dataframe is shown in the figure below:

<img src="https://user-images.githubusercontent.com/26309477/123738746-ffb15980-d8c2-11eb-847e-bd261c773817.png" width="700" /> <br>

Dataset sample statistics is given below:

<img src="https://user-images.githubusercontent.com/26309477/123740660-6ab05f80-d8c6-11eb-94b4-1962ae893157.png" width="300" /> 


### 2. Pre-processing
For Data pre-processing we have done the following steps:

- Lowercase the text
- Contraction mapping
- Removing stopwords, web-urls and tagged usernames
- Lemmatization 
- Removing non-alphabet characters
- Removing empty and duplicate tweets

Location feature cannot help in identifying whether it is a disaster tweet or non-disaster tweet. Also there are lots of empty cells. So we have removed the keyword and location column. 

Before pre-processing and after pre-processing Datapoints count statistics is given below:

| Before/After Pre-processing | Real disaster(1) Datapoints | Not real disaster(0) Datapoints | Total Datapoints |
| -------------    | -------------   |  ----------   |  ----------   
| Before  | 3271 | 4342 | 7613 |
| After | 2806 | 4033 | 6839 |

### 3. Exploratory Data Analysis:

