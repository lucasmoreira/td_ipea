#import
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report


#import data
dataset_td = pd.read_csv('trf2_td_1st.csv', delimiter = '\t')

def limpa(frase):
    try:
        frase = str(frase)
        review = re.sub('[^a-zA-Z]',' ',frase)
        review = review.lower()
        review = review.split()
        #ps = PorterStemmer()
        #review = [ps.stem(word) for word in review if not word in set (stopwords.words('portuguese'))]
        review = ' '.join(review)
        return review
    except:
        frase = "erro"
        print("erro")
        return frase

def classifica(x):
    x = str(x)
    if "improcedente" in x:
        y = x.replace('improcedente', '')
        return 1
    elif "parcialmente" in x:
        if "procedente" in x:
            y = x.replace('parcialmente', '')
            y = y.replace('procedente', '')
            return 3
        else:
            y = x
            return 0
    elif "procedente" in x:
        y = x.replace('procedente', '')
        return 2

    else:
        y = x
        return 0

def classifica2(x):
    x = str(x)
    if "improcedente" in x:
        y = x.replace('improcedente', '')
        return  y
    elif "parcialmente" in x:
        if "procedente" in x:
            y = x.replace('parcialmente', '')
            y = y.replace('procedente', '')
            return  y
        else:
            y = x
            return y
    elif "procedente" in x:
        y = x.replace('procedente', '')
        return  y

    else:
        y = x
        return y

#limpa os dados
dataset_td["texto_movimento"] = dataset_td["texto_movimento"].apply(lambda x: limpa(x))


#classifica as sentenças com regex
dataset_td["categorical"] = dataset_td["texto_movimento"].apply(lambda x: classifica(x))

dataset_td["texto_movimento"] = dataset_td["texto_movimento"].apply(lambda x: classifica2(x))


corpus = dataset_td["texto_movimento"].values.tolist()
corpus = [str(i) for i in corpus]

X_train, X_test, y_train, y_test = train_test_split(corpus, dataset_td["categorical"].values, test_size=.3, random_state=42)

cv  = TfidfVectorizer(max_features=7000, ngram_range=(1,3), max_df=0.5, min_df=2)
xtrain_tf=cv.fit_transform(X_train)


# Splitting the dataset into the Training set and Test set

#fitting classifier to training set
classifier=LinearSVC(class_weight= "balanced")
classifier.fit(xtrain_tf, y_train)


xtest_tf = cv.transform(X_test)
y_pred=classifier.predict(xtest_tf)


dftest = pd.DataFrame(data=X_test)
dftest["labeled"] = y_pred
dftest["ground_truth"] = y_test
dftest.to_csv("avaliar.csv")


#making a confusion matrix
cm=confusion_matrix(y_test,y_pred)
print(cm)
print(classification_report(y_test, y_pred))


#aplica o treino no dataset não classificado
dataset_td2 = pd.read_csv('tmp-sentencas-nao-classificadas.csv', delimiter = '\t')
dataset_td2["texto_movimento"] = dataset_td2["texto_movimento"].apply(lambda x: limpa(x))
dataset_td2["texto_movimento"] = dataset_td2["texto_movimento"].apply(lambda x: classifica2(x))
X_test = dataset_td2["texto_movimento"].values
xtest_tf = cv.transform(X_test)
y_pred=classifier.predict(xtest_tf)

dftest = pd.DataFrame(data=X_test)
dftest["labeled"] = y_pred
dftest.to_csv("avaliar10.csv")






