import nltk as nltk
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sys
from nltk.corpus import stopwords
nltk.download('stopwords')

dfTest=pd.read_csv("data/testFile.csv",sep=',')
dfTrain=pd.read_csv("data/trainFile.csv",sep=',')
dfConcat=pd.concat([dfTest,dfTrain],axis=0,ignore_index=True)
#Supprimer les lignes qui n'ont pas de titres
dfConcat=dfConcat.dropna(how='any')
listStopWords=set(stopwords.words("english"))
print(listStopWords)

