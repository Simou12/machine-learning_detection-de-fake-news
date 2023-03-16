import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
nltk.download('punkt')
nltk.download('stopwords')
import re
import unicodedata

dfTest=pd.read_csv("data/testFile.csv",sep=',')
dfTrain=pd.read_csv("data/trainFile.csv",sep=',')
dfConcat=pd.concat([dfTest,dfTrain],axis=0,ignore_index=True)

#Méthode pour la suppression d'URI
#Méthode pour la suppression d'URI
def remove_uri(text):
  # expression régulière pour détecter les URI
  pattern = r'http\S+'
  # suppression des URI
  return re.sub(pattern, '', text)

def removeStopWords(text):
  setStopWords=set(stopwords.words("english"))
  #tokenisation
  words=word_tokenize(text)
  removedStopWords=[word for word in words if word.lower() not in setStopWords]
  return ' '.join(removedStopWords)

def limmatizer(text):
  stemmer = SnowballStemmer('english')
  words=word_tokenize(text)
  wordsRemoved=[word for word in words if word.isalpha()]
  stemmed_words = [stemmer.stem(word) for word in words]
  return ' '.join(stemmed_words)


dfConcat=pd.concat([dfTest,dfTrain],axis=0,ignore_index=True)
#Suppression des doublons
dfConcat = dfConcat.drop_duplicates(subset=['title', 'text', 'our rating'], keep='first')
#supprimer les lignes qui n'ont pas de titres
dfConcat=dfConcat.dropna(subset=['title'])
listStopWords=set(stopwords.words("english"))
#Supprimer les uri dans les titres ainsi que dans les textes
dfConcat['title']=dfConcat['title'].apply(remove_uri)
dfConcat['text']=dfConcat['title'].apply(remove_uri)
#Supprimer les stop words
dfConcat['title']=dfConcat['title'].apply(removeStopWords)
dfConcat['text']=dfConcat['title'].apply(removeStopWords)
#mettre sous format utf-8
dfConcat['title'] = dfConcat['title'].str.encode('utf-8')
dfConcat['text'] = dfConcat['text'].str.encode('utf-8')
#transformer du type byte en string, du coup le b au début s'enlève
dfConcat['title'] = dfConcat['title'].str.decode('utf-8')
dfConcat['text'] = dfConcat['text'].str.decode('utf-8')
#stemmatisation
dfConcat['title'] = dfConcat['title'].apply(limmatizer)
dfConcat['text'] = dfConcat['text'].apply(limmatizer)



    





  

