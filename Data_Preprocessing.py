import re
import os
import sys
import pandas as pd
from tqdm import tqdm
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Contraction Mapping Dictionary
contraction_mapping = {"that'll":"that will","ain’t": "is not", "aren’t": "are not","can’t": "cannot", "’cause": "because", "could’ve": "could have", "couldn’t": "could not",
                          "didn’t": "did not", "doesn’t": "does not", "don’t": "do not", "hadn’t": "had not", "hasn’t": "has not", "haven’t": "have not",
                          "he’d": "he would","he’ll": "he will", "he’s": "he is", "how’d": "how did", "how’d’y": "how do you", "how’ll": "how will", "how’s": "how is",
                          "I’d": "I would", "I’d’ve": "I would have", "I’ll": "I will", "I’ll’ve": "I will have","I’m": "I am", "I’ve": "I have", "i’d": "i would",
                          "i’d’ve": "i would have", "i’ll": "i will",  "i’ll’ve": "i will have","i’m": "i am", "i’ve": "i have", "isn’t": "is not", "it’d": "it would",
                          "it’d’ve": "it would have", "it’ll": "it will", "it’ll’ve": "it will have","it’s": "it is", "let’s": "let us", "ma’am": "madam",
                          "mayn’t": "may not", "might’ve": "might have","mightn’t": "might not","mightn’t’ve": "might not have", "must’ve": "must have",
                          "mustn’t": "must not", "mustn’t’ve": "must not have", "needn’t": "need not", "needn’t’ve": "need not have","o’clock": "of the clock",
                          "oughtn’t": "ought not", "oughtn’t’ve": "ought not have", "shan’t": "shall not", "sha’n’t": "shall not", "shan’t’ve": "shall not have",
                          "she’d": "she would", "she’d’ve": "she would have", "she’ll": "she will", "she’ll’ve": "she will have", "she’s": "she is",
                          "should’ve": "should have", "shouldn’t": "should not", "shouldn’t’ve": "should not have", "so’ve": "so have","so’s": "so as",
                          "this’s": "this is","that’d": "that would", "that’d’ve": "that would have", "that’s": "that is", "there’d": "there would",
                          "there’d’ve": "there would have", "there’s": "there is", "here’s": "here is","they’d": "they would", "they’d’ve": "they would have",
                          "they’ll": "they will", "they’ll’ve": "they will have", "they’re": "they are", "they’ve": "they have", "to’ve": "to have",
                          "wasn’t": "was not", "we’d": "we would", "we’d’ve": "we would have", "we’ll": "we will", "we’ll’ve": "we will have", "we’re": "we are",
                          "we’ve": "we have", "weren’t": "were not", "what’ll": "what will", "what’ll’ve": "what will have", "what’re": "what are",
                          "what’s": "what is", "what’ve": "what have", "when’s": "when is", "when’ve": "when have", "where’d": "where did", "where’s": "where is",
                          "where’ve": "where have", "who’ll": "who will", "who’ll’ve": "who will have", "who’s": "who is", "who’ve": "who have",
                          "why’s": "why is", "why’ve": "why have", "will’ve": "will have", "won’t": "will not", "won’t’ve": "will not have",
                          "would’ve": "would have", "wouldn’t": "would not", "wouldn’t’ve": "would not have", "y’all": "you all",
                          "y’all’d": "you all would","y’all’d’ve": "you all would have","y’all’re": "you all are","y’all’ve": "you all have",
                          "you’d": "you would", "you’d’ve": "you would have", "you’ll": "you will", "you’ll’ve": "you will have",
                          "you’re": "you are", "you’ve": "you have","ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not",
                          "didn't": "did not", "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not",
                          "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",
                          "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would",
                          "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would",
                          "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam",
                          "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have",
                          "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock",
                          "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have",
                          "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is",
                          "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as",
                          "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would",
                          "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have",
                          "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have",
                          "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are",
                          "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",
                          "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is",
                          "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have",
                          "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have",
                          "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all",
                          "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have",
                          "you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have",
                          "you're": "you are", "you've": "you have","n't":'not'}



def data_preprocess(tweets):
    """ data_preprocess() function takes a list of strings as input
        and returns the pre-processed strings list
        Input : ['#Spotlight Take Me To Paradise by Arsonist MC #WNIAGospel http://t.co/1he4UfaWZm @arsonistmusic http://t.co/BNhtxAEZMM']
        Output : ['spotlight take paradise arsonist wniagospel']
    """
    clean_data=[]

    # Iterating through each tweet
    for line in tqdm(tweets):
        x=line.lower()              #lower-casing the words
        word_list=[]

        # Iterating through each word in the tweet
        for word in x.split():      # contraction mapping
            if word in contraction_mapping:
                word_list.append(contraction_mapping[word])
            else:
                word_list.append(word)

        x=' '.join(word_list)
        x=re.sub("'s",' ',x)                    # removing 's, Example : it's --> it
        x=re.sub('\n',' ',x)                      # removing newline character
        x=re.sub(r'http.*?\s|http.*',' ',x)       # removing web urls like - http://t.co/lhyxeohy6c
        x=re.sub(r'@.*?\s|@(.*)?',' ',x)          # removing tagged usernames like - @bbcmtd
        x=re.sub(r'\([^a-z]*?\)',' ',x)           # removing text like : (08/06/15)
        x=re.sub(r'\[[^a-z]*?\]',' ',x)           # removing text like : [08/06/15]
        x=re.sub(r'\{[^a-z]*?\}',' ',x)           # removing text like : {08/06/15}
        x=re.sub(r'&[a-z]*;*',' ',x)              # removing keywords like : &amp &amp;
        x=re.sub(r'[^a-z]',' ',x)                 # removing non-alphabetic characters

        x=re.sub(r'a{3,}','aa',x)                 #
        x=re.sub(r'b{3,}','bb',x)                 #
        x=re.sub(r'c{3,}','cc',x)                 #
        x=re.sub(r'd{3,}','dd',x)                 #
        x=re.sub(r'e{3,}','ee',x)                 #
        x=re.sub(r'f{3,}','ff',x)                 #
        x=re.sub(r'g{3,}','gg',x)                 #
        x=re.sub(r'h{3,}','hh',x)                 #
        x=re.sub(r'i{3,}','ii',x)                 # correcting words like : coooooool  -->  cool
        x=re.sub(r'j{3,}','jj',x)                 #
        x=re.sub(r'k{3,}','kk',x)                 #
        x=re.sub(r'l{3,}','ll',x)                 #
        x=re.sub(r'm{3,}','mm',x)                 #
        x=re.sub(r'n{3,}','nn',x)                 #
        x=re.sub(r'o{3,}','oo',x)                 #
        x=re.sub(r'p{3,}','pp',x)                 #
        x=re.sub(r'q{3,}','qq',x)                 #
        x=re.sub(r'r{3,}','rr',x)                 #
        x=re.sub(r's{3,}','ss',x)                 #
        x=re.sub(r't{3,}','tt',x)                 #
        x=re.sub(r'u{3,}','uu',x)                 #
        x=re.sub(r'v{3,}','vv',x)                 #
        x=re.sub(r'w{3,}','ww',x)                 #
        x=re.sub(r'x{3,}','xx',x)                 #
        x=re.sub(r'y{3,}','yy',x)                 #
        x=re.sub(r'z{3,}','zz',x)                 #

        x=re.sub(r'\s+',' ',x)                    # replacing multiple spaces with single spaces
        x=x.strip()                               # Removing leftmost and rightmost spaces
        x=lemmatization(x)                        # Lemmatizing the string
        x=stopword_removal(x)                     # Removing stopwords
        clean_data.append(x)

    return clean_data

def lemmatization(tweet):
    """ lemmatization() function takes a string as input 
        and returns the string after doing lemmatization
        Input : "who makes these"
        Output : "who make these"
    """
    wordnet_lemmatizer = WordNetLemmatizer()
    word_list = []

    # Iterating through each word in the tweet
    for word in tweet.split():
        word = wordnet_lemmatizer.lemmatize(word,'v')
        word = wordnet_lemmatizer.lemmatize(word,'n')
        word_list.append(word)
    return " ".join(word_list)

def stopword_removal(tweet):
    """ stopword_removal() function takes a string as input 
        and returns the string after removing stopwords and words having length less <= 2 except the word 'no' and 'not'
        Input : "on town of salem i just melted ice cube bc im the arsonist :D"
        Output : "town salem melted ice cube arsonist"
    """
    non_stopwords = {'no', 'not'}
    stopword = stopwords.words('english')
    word_list = []

    # Iterating through each word in the tweet
    for word in tweet.split():
        if word in non_stopwords:
            word_list.append(word)
        else:
            if len(word) > 2 and word not in stopword:
                word_list.append(word)
    return " ".join(word_list)

file_name = sys.argv[1]                                      # Getting File name
print("File Name : {}".format(file_name))
try:
    df = pd.read_csv(file_name)                              # Reading .CSV file
    df.drop("id",axis=1, inplace=True)                       #
    df.drop("keyword",axis=1, inplace=True)                  # Removing Extra Columns
    df.drop("location",axis=1, inplace=True)                 # 
    #df.drop("target",axis=1, inplace=True)
    print("\nPreprocessing Dataset\n")
    df["text"] = data_preprocess(df["text"])
    df.drop_duplicates(inplace=True)                         # Removing duplicate rows
    print("\nPreprocessing Completed\n")
    output_file = "clean_" + os.path.basename(file_name)     # output .csv file name
    df.to_csv(output_file,index=False)                       # Saving output .csv file
    print("Preprocessed data saved to the file : {}".format(output_file))
    
except Exception as e:
    print("File doesn't exists. Raised error : {}".format(e))