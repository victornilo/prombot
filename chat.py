#### Importing necessary libraries
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from nltk_utils import remove_punctuations, remove_stopwords, Stemming, Lemmatize, Recreate
from sklearn.naive_bayes import MultinomialNB
from prep import Cleaning
import random
import json

with open('iPromote/intents.json', 'r') as json_data:
    intents = json.load(json_data)

#### NLP Pipeline Reference: https://www.analyticsvidhya.com/blog/2022/06/an-end-to-end-guide-on-nlp-pipeline/
# NLP Pipeline is a set of steps followed to build an end to end NLP software.
# Data Collection
# Text Cleaning
# Pre Processing
# Feature Engineering
# Modeling
# Evaluation
# Deployment
# Monitoring and Model Updating

#### Creating a ChatBot using basic ML Algorithm
# Reference: https://medium.com/@divalicious.priya/creating-a-chatbot-using-the-basic-ml-algorithms-part-1-70f6af52c2e3

##### STEP 1: Data Acquisition
# Read Data
data=pd.read_csv("iPromote/Data.csv")

#### STEP 2: Text Preprocessing
# Types of Text Preprocessing
# A. Text Cleaning – We do HTML tag removing, emoji handling, Spelling checker, etc.
# B. Basic Preprocessing — We do tokenization (word or sent tokenization, stop word removal, removing digit, lower casing.
# C. Advance Preprocessing — We do POS tagging, Parsing, and Coreference resolution.

# TOKENIZATION
# Tokenization is used to split a large sample of text or sentences into words.
#data["question_punctuation_removed"]=data["question"].apply(tk_function)
#print (data["question_punctuation_removed"])
 
data["question_punctuation_removed"]=data["question"].apply(remove_punctuations)
#print (data["question_punctuation_removed"])

#data["question_stopword_removed"]=data["question_punctuation_removed"].apply(remove_stopwords)
#print (data["question_stopword_removed"])

#data["question_stemmed"]=data["question_stopword_removed"].apply(Stemming)
#print (data["question_stemmed"])

#data["question_lemmatized"]=data["question_stopword_removed"].apply(Lemmatize)
#print (data["question_lemmatized"])

#data["modified_sentence"]=data["question_stemmed"].apply(Recreate)
#print (data["modified_sentence"])	

data["question_stemmed"]=data["question_punctuation_removed"].apply(Stemming)

data["modified_sentence"]=data["question_stemmed"].apply(Recreate)
print (data["modified_sentence"])

#### STEP 3: Feature Engineering
# Feature Engineering means converting text data to numerical data.
# It is required to convert text data to numerical data, because our machine learning model 
# doesn’t understand text data. This step is also called Feature extraction from text.

# Let's change the sentence into a bag of word (BOW) model
# BOW is one of the most used text vectorization techniques.
# A bag-of-words is a representation of text that describes the occurrence of words within a document.
# Specially used in the Text Classification task.
# We can directly use CountVectorizer class by Scikit-learn.

bow_vectorizer = CountVectorizer()

# Creating training data
X = bow_vectorizer.fit_transform(data["modified_sentence"]).toarray()
Y = data["answer"]

#### STEP 4: Modelling/Model Building
# In the modeling step, we try to make a model based on data.
# We can use multiple approaches (Heuristic Approach, Machine Learning Approach, Deep Learning Approach, or Cloud API)
# to build the model based on the problem statement.

# Create Classifier
clf2 = MultinomialNB()
clf2.fit(X, Y)

#### STEP 5: Model Evaluation
# In the model evaluation, we can use two metrics Intrinsic evaluation and Extrinsic evaluation.
# Intrinsic evaluation uses multiple metrics to check the model such as Accuracy, Recall, Confusion Metrics, Perplexity, etc.
# Extrinsic evaluation is done after deployment. This is the business-centric approach.
# Checking Accuracy of the Classifier is in another program accuracy.py

#### STEP 6: Deployment
# We have to deploy our model for the users.
# Users can use this model.
# Deployment has three stages deployment, monitoring, and retraining or model update.
# A. Deployment – model deploying (on the cloud) for users.
# B. Monitoring – we have to watch the model continuously. Here we have to create a dashboard to show evaluation metrics.
# C. Update- Retrain the model on new data and again deploy.

# Ready to classify new input
# Create functions to take user input, pre-process the input, predict the class, and get the response.
  
def generate_response(sentence):
    input_vector=bow_vectorizer.transform([Cleaning(sentence)])
    #print(input_vector)
    predicted=clf2.predict(input_vector)
    predicted=Recreate(predicted)
    #answer=generate_answer(predict2)
    #return answer
    
    for intent in intents['intents']:
            counter=0
            if predicted == intent["tag"]:
                counter=counter+1
                return random.choice(intent['responses'])
    if counter==0:
      return "Please try rephrasing or asking another question."

if __name__ == "__main__":
    print("Let's chat! (type 'quit' to exit)")
    while True:
        # sentence = "do you use credit cards?"
        sentence = input("You: ")
        if sentence == "quit":
            break

        resp = generate_response(sentence)
        print(resp)