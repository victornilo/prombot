import re
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer

##### Now let's pre process/ clean our data
# Data preprocessing is the manipulation or dropping of data before it is used in order to ensure 
# or enhance performance, and is an important step in the data mining process.

### Remove Punctuations and change words to lower case
# In this program, punctuations do not convey any meaning.
def remove_punctuations(text):    
    words=[word.lower() for word in text.split()] 
    #print ("words after step 1",words)
    words=[w for word in words for w in re.sub(r'[^\w\s]',' ',word).split()]    
    return words

### Remove StopWords
# Stopwords are words usually support the main words.
# We will remove them since they do not convey much information for data processing.
# Stop words are only for sentence formation.
# In the meaning of the sentence, stop words are not important.
from nltk.corpus import stopwords
stop = set(stopwords.words('english'))
#print (stop)

def remove_stopwords(text):
	modified_word_list=[word for word in text if word not in stop]
	return modified_word_list

### Stemming of Words
# Stemming means mapping a group of words to the same stem by removing prefixes or suffixes
# without giving any value to the “grammatical meaning” of the stem formed after the process.
# Types of Stemmer — It is an algorithm to do stemming
# A. Porter Stemmer — specific for the English language
# B. Snowball Stemmer — used for multiple languages
# C. Lancaster Stemmer
st=PorterStemmer()
def Stemming(text):
	stemmed_words=[st.stem(word) for word in text] 
	return stemmed_words

### Lemmatizing of Words
# Lemmatization also does the same thing as stemming and tries to bring a word to its base form,
# but unlike stemming it does keep into account the actual meaning of the base word.
# In Lemmatization we search words in wordnet.
wl = WordNetLemmatizer()
def Lemmatize(text):
    lemmatized_words=[wl.lemmatize(word, pos='v') for word in text]
    return lemmatized_words

### Recreating the sentence
def Recreate(text):
	word=" ".join(text)
	return word