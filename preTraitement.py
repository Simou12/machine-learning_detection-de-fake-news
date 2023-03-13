import nltk
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import re
import unicodedata
nltk.download('stopwords')
from nltk.corpus import stopwords


dfTest=pd.read_csv("data/testFile.csv",sep=',')
dfTrain=pd.read_csv("data/trainFile.csv",sep=',')
dfConcat=pd.concat([dfTest,dfTrain],axis=0,ignore_index=True)

#Supprimer les lignes qui n'ont pas de titres
dfConcat=dfConcat.dropna(how='any')


listStopWords=set(stopwords.words("english"))
for titre in dfConcat['title']:
    #print(titre)
    mots = titre.split()
    #Supprimer les stopwords
    filtered_mots = [mot for mot in mots if mot.lower() not in listStopWords]
    texte_sans_stopwords = ' '.join(filtered_mots)
    #Supprimer les URLs
    texte_sans_url = re.sub(r'http[s]?://\S+', '', texte_sans_stopwords)
    print(texte_sans_url)
    #convertir les données en UTF8
    chaine=unicodedata.normalize('NFKD', texte_sans_url).encode('ascii','ignore')
    
    


    





  

