from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import seaborn as sns
import pandas as pd
import numpy as np
### Load the Data set
df = pd.read_csv("diabetes.csv")

from sklearn.model_selection import train_test_split
X = df[['BloodPressure', 'Age', 'Insulin', 'Glucose', 'BMI']]
y = df['Outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

### Train the model
clf = DecisionTreeClassifier(random_state=0, criterion='entropy', max_depth=3, min_samples_split=2, min_samples_leaf=1)
clf.fit(X_train, y_train)

### Qualitative Evaluation
y_preds = clf.predict(X_test)
print("Accuracy %.4f" % accuracy_score(y_test, y_preds))
print("Precision %.4f" % precision_score(y_test, y_preds))
print("Recall %.4f" % recall_score(y_test, y_preds))
print("F1 %.4f" % f1_score(y_test, y_preds))

### Display the current tree
plot_tree(clf, feature_names = list(X.columns), class_names = ['No Diabetes', 'Diabetes'],filled = True)

