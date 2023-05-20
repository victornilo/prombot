from sklearn.metrics import confusion_matrix
from prep import Cleaning
from collections import Counter
import warnings
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import classification_report
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from nltk_utils import remove_punctuations, remove_stopwords, Stemming, Recreate

data=pd.read_csv("A-CB/Data.csv")

data["question_punctuation_removed"]=data["question"].apply(remove_punctuations)
#print (data["question_punctuation_removed"])

data["question_stopword_removed"]=data["question_punctuation_removed"].apply(remove_stopwords)
#print (data["question_stopword_removed"])

data["question_stemmed"]=data["question_stopword_removed"].apply(Stemming)
#print (data["question_stemmed"])

data["modified_sentence"]=data["question_stemmed"].apply(Recreate)
#print (data["modified_sentence"])	

bow_vectorizer = CountVectorizer()
X = bow_vectorizer.fit_transform(data["modified_sentence"]).toarray()
Y = data["answer"]

clf2 = MultinomialNB()
clf2.fit(X, Y)

#clf2 = DecisionTreeClassifier()
#clf2.fit(X, Y)

def Predict(text):
    input_vector=bow_vectorizer.transform([Cleaning(text)])
    predict2=clf2.predict(input_vector)    
    final_predict=[]
    final_predict=list(predict2)
    final_predict = Counter(final_predict)
    return final_predict.most_common(1)[0][0]

X_test=["What's up","I hate these kinds of things","are you a robot?","how is it going"]
Y_test=["greeting","hate","whoareyou","greeting"]

Y_pred=[]
for i in X_test:
    prediction=Predict(i)
    Y_pred.append(prediction)

#### Confusion Matrix Reference: https://towardsdatascience.com/understanding-confusion-matrix-a9ad42dcfd62
# Cnf is performance measurement for machine learning classification problem where output can be 
# two or more classes. It is a table with 4 different combinations of # predicted and actual values.
# Extremely useful for measuring Recall, Precision, Specificity, Accuracy, and AUC-ROC curves.

# Recall=TP/TP+FN
# Recall is explained by saying, from all the positive classes, how many we predicted correctly.
# Recall should be high as possible.

# Precision=TP/TP+FP 
# Precision is explained by saying, from all the classes we have predicted as positive, how many 
# are actually positive. Precision should be high as possible.

# Accuracy - From all the classes (positive and negative), how many of them we have predicted 
# correctly. Accuracy should be high as possible.

# F-measure
# It is difficult to compare two models with low precision and high recall or vice versa.
# So to make them comparable, we use F-Score.
# F-score helps to measure Recall and Precision at the same time.
# It uses Harmonic Mean in place of Arithmetic Mean by punishing the extreme values more.

cnf_matrix = confusion_matrix(Y_test,Y_pred)
print (cnf_matrix)

print("Multinomial Naive Bayes model accuracy(in %):", metrics.accuracy_score(Y_test, Y_pred)*100)
warnings.filterwarnings("ignore")
print(classification_report(Y_test, Y_pred))