import pandas as pd
from sklearn.model_selection import train_test_split
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

# Initialize and train Naive Bayes classifier
nb_classifier = GaussianNB()
nb_classifier.fit(X_train, y_train)

# Evaluate Naive Bayes
nb_predictions = nb_classifier.predict(X_test)
print("Naive Bayes Classifier:")
print("Accuracy:", accuracy_score(y_test, nb_predictions))
print(classification_report(y_test, nb_predictions))

# Allow user to input new data for classification
def classify_new_data():
    print("\nEnter new customer data for classification:")
    region = input("Region (Lisbon=0, Oporto=1, Other=2): ")
    fresh = input("Fresh: ")
    milk = input("Milk: ")
    grocery = input("Grocery: ")
    frozen = input("Frozen: ")
    detergents_paper = input("Detergents_Paper: ")
    delicassen = input("Delicassen: ")
    
    # Create a DataFrame from user input
    new_data = pd.DataFrame([[int(region), int(fresh), int(milk), int(grocery), int(frozen), int(detergents_paper), int(delicassen)]],
                            columns=['Region', 'Fresh', 'Milk', 'Grocery', 'Frozen', 'Detergents_Paper', 'Delicassen'])
    
    # Predict the channel
    prediction = nb_classifier.predict(new_data)
    channel = "HoReCa" if prediction[0] == 0 else "Retail"
    print(f"The predicted channel for the given data is: {channel}")

# Call the function to classify new data
classify_new_data()
