# Limpa e prepara os dados de texto das sentencas

import pandas as pd
import ssl
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


#nao lembro pra que serve
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

#import data
dataset_td = pd.read_csv('trf2_td_1st.csv', delimiter = '\t')

def limpa(frase):
    try:
        frase = str(frase)
        review = re.sub('[^a-zA-Z]',' ',frase)
        review = review.lower()
        review = review.split()
        ps = PorterStemmer()
        review = [ps.stem(word) for word in review if not word in set (stopwords.words('portuguese'))]
        review = ' '.join(review)
        return review
    except:
        frase = "erro"
        print("erro")
        return frase

#calcula o tempo levado
import time
start = time.time()

#aplica a funcao e prepara o texto
dataset_td["texto_movimento_novo"] = dataset_td["texto_movimento"].apply(lambda x: limpa(x))

#imprime o tempo gasto
end = time.time()
print(end - start)

#exporta os dados para serem usados na etapa 02
dataset_td.to_csv("dados_stemizados.csv", sep = ';', index = False)
