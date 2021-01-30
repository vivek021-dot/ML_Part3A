import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import pickle

df = pd.read_csv("churning.csv")
# we use only the first 2 variables as features
# account length,number vmail messages
x = df.iloc[:, [0, 1]]
y = df.iloc[:,-1] # classification
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
classifier = AdaBoostClassifier(n_estimators=10, learning_rate=0.8, random_state=0, algorithm='SAMME')
classifier.fit(x, y)
y_pred = classifier.predict(x_test)
cm = confusion_matrix(y_test, y_pred)

# w=write, b=binary
with open("my_ml_model1.ml", 'wb') as f:
    pickle.dump(classifier, f)

print(cm)