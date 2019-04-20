#classifica com NB e plota resultados
import pandas as pd

#importing dataset
result = pd.read_csv('categorical.csv', delimiter = ',')

#taking the entire text moviment and converting to string
corpus = result["texto_movimento_novo"].values.tolist()
corpus = [str(i) for i in corpus]


from sklearn.feature_extraction.text import TfidfVectorizer

max_features = 1500
cv  = TfidfVectorizer(max_features=10000)
X=cv.fit_transform(corpus)
y=result["categorical"].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 42)


#fitting classifier to training set
from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression()
classifier.fit(X_train,y_train)


#predicting the test set result
y_pred=classifier.predict(X_test)


#making a confusion matrix
from sklearn.metrics import confusion_matrix, classification_report
cm=confusion_matrix(y_test,y_pred)
print(cm)
print(classification_report(y_test, y_pred))

'''
Plotar resultado
'''
# y_true = ["cat", "ant", "cat", "cat", "ant", "bird"]
# y_pred = ["ant", "ant", "cat", "cat", "ant", "cat"]
# print(confusion_matrix(y_true, y_pred))
#
#
# # Visualising the Test set results
# from matplotlib.colors import ListedColormap
# X_set, y_set = X_test, y_test
#
# X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
#                      np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
#
# plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
#              alpha = 0.75, cmap = ListedColormap(('red', 'green')))
# plt.xlim(X1.min(), X1.max())
# plt.ylim(X2.min(), X2.max())
# for i, j in enumerate(np.unique(y_set)):
#     plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
#                 c = ListedColormap(('red', 'green'))(i), label = j)
# plt.title('Naive Bayes (Test set)')
# plt.xlabel('Age')
# plt.ylabel('Estimated Salary')
# plt.legend()
# plt.show()
