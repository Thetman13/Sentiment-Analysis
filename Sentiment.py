#Tyler Porter
#A Native Bayes classifier to determine the sentiment of a given statement.
#This will use Native Bayes classification to determine the sentiment of a statement

#Be sure this is in the same folder as "trainingdata.csv"

#imports
import numpy as np
import pandas as pd
import re
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import train_test_split
from sklearn import metrics


#Global variables
#labels holds the sentiment value of each sentence
labels = []

#sentences holds the text sentencews to be read
sentences = []

#bag of words
wordlist = [[], []]


totalpositive = 0
totalnegative = 0
Phappy = 0 #Probablility a tweet is happy. num of positive tweets / total tweets
Psad = 0 # Probability a tweet is sad. num of negative tweets / total tweets

trainingnum = 500000

model = []

vocabulary = [] #This holds all vocabulary used for the model to refit custom data testing


#Methods
def Intake():
    #This method takes in all data and stores it into appropraite formats
    global labels, sentences, trainingnum
    #without the encoding line it will not read lines from csv
    #nrows limited to 10000 for printing and speed until training model
    filepath = 'trainingdata.csv' #MAY NEED CHANGED take input from user for filepath
    data = pd.read_csv(filepath, error_bad_lines=False, encoding="ISO-8859-1", nrows=trainingnum)
    #allocates sentiment ratings and sentences to 2 arrays
    labels = data.iloc[:, 1]
    sentences = data.iloc[:, 2]


def Organize():
    #This method sorts tweets into good and bad
    global totalpositive
    global totalnegative
    for i in range(len(labels)):
        if (labels[i] == 0):
            totalnegative = totalnegative + 1
        elif (labels[i] == 1):
            totalpositive = totalpositive + 1
    print("Total Positive = ")
    print(totalpositive)
    print("Total Negative = ")
    print(totalnegative)
    Phappy = totalpositive / (totalpositive + totalnegative)
    Psad = totalnegative / (totalpositive + totalnegative)
    print("Probablilty Happy :", Phappy, "Probablilty Sad :", Psad)

    #intakes a sentence to be tested
def Custom_Test():
    global model
    global vocabulary
    test_matrix = np.zeros([len(vocabulary)])
    text = input("\nType a sentence you would like the model to analyze the sentiment of.\n")
    #Cleans text input
    text = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','URL', text) #replaces urls with blanket 'URL'
    text = re.sub('@[^\s]+','USER', text) #replaces usernames with 'USER'
    text = text.lower().replace("ё", "е")
    text = re.sub('[^a-zA-Zа-яА-Я]+', ' ', text)
    text = re.sub(' +',' ', text)
    text = re.sub(r'\W*\b\w{1,2}\b', "", text).strip()
    text_array = text.split( )
    #Makes a matrix in same shape and meaning as the vecotorizer
    #makes value the number of times a word appears
    for i in range(len(vocabulary)):
        for j in range(len(text_array)):
            if (vocabulary[i]==text_array[j]):
                test_matrix[i] = test_matrix[i] + 1
    test_matrix = test_matrix.reshape(1, -1)
    result = model.predict(test_matrix)
    if (result):
        print("-> Positively-Worded")
    else:
        print("-> Negatively-Worded")

    temp = input("To test another sentence input 'Y'.\nTo exit input 'N'.\n")
    repeat = False
    if (temp == "Y"):
        repeat = True
    if (repeat):
        Custom_Test()
    return

def findProbabilities(data):
    global labels
    print("Beginning build of model, this may take a moment...")
    #test size / sets aside X% of data for testing, (1-X%) for training
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(data, labels, test_size = 0.25, random_state = 2)
    model = BernoulliNB()
    model.fit(Xtrain, Ytrain)
    accuracy = metrics.accuracy_score(model.predict(Xtest), Ytest)
    print("Model accuracy = " + str('{:5.2f}'.format(accuracy * 100)) + '%')

    return model


def BagOfWords():
    #builds the vocabulary list from each line and the frequency of words
    global wordlist
    global vocabulary
    vocab = CountVectorizer()
    temp = vocab.fit_transform(sentences)
    #vocab.fit(sentences)
    #wordlist = temp.toarray()
    #print("Vocabulary:", vocab.get_feature_names())
    vocabulary = vocab.get_feature_names()
    print("Size of vocabulary:", len(vocabulary))

    #wordlist is represented as (a, b) c
    #where a is the sentence, b is the index of a given word, (made when vocab.fit(sentences))
    #and c is the number of times the word appears in sentence a


    #outputs 2d array
    #i = word index
    #j = each individual tweet
    #temp[i][j] The number of times word i appears in tweet j
    return (temp)

def Clean(sentences):
    #This method removes non alphanumeric characters, whitespace, and words of length 2 or less
    #normally words of 3 or less are removed, but i feel words like "sad", "mad" and others of
    #length 3 appear often enough and are important enough to be counted.
    for i in range(len(sentences)):

        text = sentences[i]
        text = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','URL', text) #replaces urls with blanket 'URL'
        text = re.sub('@[^\s]+','USER', text) #replaces usernames with 'USER'
        text = text.lower().replace("ё", "е")
        text = re.sub('[^a-zA-Zа-яА-Я]+', ' ', text)
        text = re.sub(' +',' ', text)
        text = re.sub(r'\W*\b\w{1,2}\b', "", text).strip()

        sentences[i] = text
    return

#Main
print("\nBeginning read-in from excel file. If n >= 100,000 this may take a moment\n")
Intake()
print("\nCleaning and vectorizing strings for prep into vocabulary\n")
Clean(sentences)
print("\nCalculating total positive and total negative tweets for preliminiary data\n")
Organize()
rawarray = BagOfWords()
model = findProbabilities(rawarray)

Custom_Test()
