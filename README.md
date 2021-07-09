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

#### Running data pre-process script:

```sh
python3 Data_Preprocessing.py train.csv
```
#### Output Terminal:

<img src="https://user-images.githubusercontent.com/26309477/123936112-83963f00-d9b2-11eb-8676-256bb29f95cf.png" width="900" /> 

#### Examples:

| Before Pre-processing | After Pre-processing |
| -------------    | ------------- 
| @etribune  US Drone attack kills 4-suspected militants in North Waziristan @AceBreakingNews https://t.co/jB038rdFAK | drone attack kill suspect militant north waziristan |
| What a feat! Watch the #BTS of @kallemattson's incredible music video for #Avalanche: https://t.co/3W6seA9tuv ???? | feat watch bts incredible music video avalanche |
| Dragon Ball Z: Battle Of Gods (2014) - Rotten Tomatoes http://t.co/jDDNhmrmMJ via @RottenTomatoes | dragon ball battle god rotten tomato via |
| Metal Cutting Sparks Brush Fire In Brighton: A brush fire that was sparked by a landowner cutting metal burned 10Û_ http://t.co/rj7m42AtWS | metal cut spark brush fire brighton brush fire spark landowner cut metal burn |



Before pre-processing and after pre-processing Datapoints count statistics is given below:

| Before/After Pre-processing | Real disaster(1) Datapoints | Not real disaster(0) Datapoints | Total Datapoints |
| -------------    | -------------   |  ----------   |  ----------   
| Before  | 3271 | 4342 | 7613 |
| After | 2806 | 4033 | 6839 |

### 3. Exploratory Data Analysis:
#### Complete Data Word Count Distribution:

<img src="https://user-images.githubusercontent.com/26309477/123932520-2947af00-d9af-11eb-8fef-87f6ae2d0821.jpg" width="900" /> 

#### Class-wise Data Word Count Distribution:

<img src="https://user-images.githubusercontent.com/26309477/123932756-5f852e80-d9af-11eb-850a-ff03ae72fec9.jpg" width="900" /> 

#### Key Observations:

- 50% tweets have less than 9 words.
- There are very less tweets that have more than 18 words.
- 50 % of the Disaster tweets have 7 to 11 words.
- 50 % of the Non-Disaster tweets have 5 to 11 words.
- Disaster class has very less tweets having words more than 17.
- Minimum tweet length is 1 and maximum tweet length is 21.
- There are total 11095 unique words.

#### Top 20 Most Frequent words:

{'not': 639, 'get': 407, 'like': 386, 'fire': 339, 'no': 254, 'one': 202, 'bomb': 191, 'say': 188, 'would': 184, 'new': 183, 
 'via': 182, 'people': 181, 'news': 177, 'burn': 175, 'time': 174, 'make': 173, 'emergency': 154, 'video': 152, 'flood': 151, 'build': 151}
 
#### Word Cloud:

<img src="https://user-images.githubusercontent.com/26309477/123934635-1afa9280-d9b1-11eb-9526-82523cf53442.jpg" width="400" /> 

### 4. Feature Extraction:
We have extracted features using Count Vector, Binary Bows, TF-IDF Vector, Bi-Gram Vector, Average Word2Vec and TF-IDF weighted Word2Vec.

Based on the experiments, we found that we are getting good results using features: TF-IDF Vector, Average Word2Vec and TF-IDF weighted Word2Vec.

For TF-IDF Vector, Vocabulary Size is : 110095.

For Word2Vec model, vector size is 64. Based on experiments, we found that training Word2Vec model for 100 epochs produces good results.

### 5. Model Training:

#### 1. K - nearest neighbors Algorithm:

KNN Algorithms Training & Testing Details are given below: 

Features | Best K | F1 - Score 
-------------    | -------------   |  ---------- 
TF-IDF Vector |      41           |0.77750
Average Word2Vec  | 7 | 0.76371 
 TF-IDF weighted Word2Vec | 7 | 0.76647
    
    
    
    
    
