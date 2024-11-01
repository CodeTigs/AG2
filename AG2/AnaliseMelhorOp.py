import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report

# Load the CSV file into a pandas DataFrame
file_path = 'D:\AG2/Wholesale customers data.csv'
data = pd.read_csv(file_path)

# Mapping the values based on the provided image
channel_mapping = {"HoReCa": 0, "Retail": 1}
region_mapping = {"Lisbon": 0, "Oporto": 1, "Other": 2}

# Assuming the 'Channel' and 'Region' columns are strings that need to be converted
data['Channel'] = data['Channel'].replace(channel_mapping).astype('int64')
data['Region'] = data['Region'].replace(region_mapping).astype('int64')

# Reorder the columns
data = data[['Region', 'Fresh', 'Milk', 'Grocery', 'Frozen', 'Detergents_Paper', 'Delicassen', 'Channel']]

# Split the data into training and testing sets (80% training, 20% testing)
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Separate features and target variable
X_train = train_data.drop('Channel', axis=1)
y_train = train_data['Channel']
X_test = test_data.drop('Channel', axis=1)
y_test = test_data['Channel']

# Initialize classifiers
dt_classifier = DecisionTreeClassifier(random_state=42)
knn_classifier = KNeighborsClassifier()
mlp_classifier = MLPClassifier(max_iter=500, random_state=42)
nb_classifier = GaussianNB()

# Train and evaluate Decision Tree
dt_classifier.fit(X_train, y_train)
dt_predictions = dt_classifier.predict(X_test)
print("Decision Tree Classifier:")
print("Accuracy:", accuracy_score(y_test, dt_predictions))
print(classification_report(y_test, dt_predictions))

# Train and evaluate k-Nearest Neighbors
knn_classifier.fit(X_train, y_train)
knn_predictions = knn_classifier.predict(X_test)
print("k-Nearest Neighbors Classifier:")
print("Accuracy:", accuracy_score(y_test, knn_predictions))
print(classification_report(y_test, knn_predictions))

# Train and evaluate Multilayer Perceptron
mlp_classifier.fit(X_train, y_train)
mlp_predictions = mlp_classifier.predict(X_test)
print("Multilayer Perceptron Classifier:")
print("Accuracy:", accuracy_score(y_test, mlp_predictions))
print(classification_report(y_test, mlp_predictions))

# Train and evaluate Naive Bayes
nb_classifier.fit(X_train, y_train)
nb_predictions = nb_classifier.predict(X_test)
print("Naive Bayes Classifier:")
print("Accuracy:", accuracy_score(y_test, nb_predictions))
print(classification_report(y_test, nb_predictions))
