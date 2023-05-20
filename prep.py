import random
from nltk_utils import remove_punctuations, remove_stopwords, Stemming, Recreate

def Cleaning(text):
    text_punctuation_removed=remove_punctuations(text)
    #text_stopword_removed=remove_stopwords(text_punctuation_removed)
    text_stemmed=Stemming(text_punctuation_removed)
    final_text=Recreate(text_stemmed)
    #print (final_text)
    return final_text