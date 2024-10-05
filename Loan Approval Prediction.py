import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load the training dataset
train_data = pd.read_csv('..\Training Dataset.csv')

# Check for missing values
print(train_data.isnull().sum())

# Fill missing values with appropriate strategies
train_data['LoanAmount'].fillna(train_data['LoanAmount'].median(), inplace=True)
train_data['Loan_Amount_Term'].fillna(train_data['Loan_Amount_Term'].mode()[0], inplace=True)
train_data['Credit_History'].fillna(train_data['Credit_History'].mode()[0], inplace=True)
train_data['Gender'].fillna(train_data['Gender'].mode()[0], inplace=True)
train_data['Married'].fillna(train_data['Married'].mode()[0], inplace=True)
train_data['Dependents'].fillna(train_data['Dependents'].mode()[0], inplace=True)
train_data['Self_Employed'].fillna(train_data['Self_Employed'].mode()[0], inplace=True)

# Drop the Loan_ID column as it is not relevant for clustering
train_data.drop('Loan_ID', axis=1, inplace=True)

# Encode categorical variables
label_encoders = {}
for column in train_data.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    train_data[column] = le.fit_transform(train_data[column])
    label_encoders[column] = le

# Standardize the data
scaler = StandardScaler()
scaled_train_data = scaler.fit_transform(train_data.drop('Loan_Status', axis=1))

# Determine the optimal number of clusters using the elbow method
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(scaled_train_data)
    wcss.append(kmeans.inertia_)

# Plot the elbow graph
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), wcss, marker='o', linestyle='-')
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# Choose the optimal number of clusters (based on the elbow plot)
optimal_clusters = 3

# Apply KMeans with the optimal number of clusters
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
train_data['Cluster'] = kmeans.fit_predict(scaled_train_data)

# Analyze the characteristics of each cluster
cluster_analysis = train_data.groupby('Cluster').mean()
print(cluster_analysis)

# Plot the clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(data=train_data, x='ApplicantIncome', y='LoanAmount', hue='Cluster', palette='viridis')
plt.title('Clusters of Loan Applicants')
plt.xlabel('Applicant Income')
plt.ylabel('Loan Amount')
plt.show()

# Add a column for loan approval status in the dataset
train_data['Loan_Status'] = label_encoders['Loan_Status'].inverse_transform(train_data['Loan_Status'])

# Analyze loan approval status within each cluster
cluster_approval = train_data.groupby(['Cluster', 'Loan_Status']).size().unstack().fillna(0)
print(cluster_approval)

# Predict loan approval based on the most frequent loan status in each cluster
def predict_loan_approval(applicant_data):
    applicant_scaled = scaler.transform([applicant_data])
    cluster = kmeans.predict(applicant_scaled)[0]
    most_common_status = cluster_approval.loc[cluster].idxmax()
    return most_common_status

# Example prediction
example_applicant = train_data.iloc[0].drop(['Loan_Status', 'Cluster']).values
predicted_status = predict_loan_approval(example_applicant)
print(f'Predicted Loan Status for the example applicant: {predicted_status}')

# Load the test dataset
test_data = pd.read_csv('..\Test Dataset.csv')

# Fill missing values with appropriate strategies
test_data['LoanAmount'].fillna(test_data['LoanAmount'].median(), inplace=True)
test_data['Loan_Amount_Term'].fillna(test_data['Loan_Amount_Term'].mode()[0], inplace=True)
test_data['Credit_History'].fillna(test_data['Credit_History'].mode()[0], inplace=True)
test_data['Gender'].fillna(test_data['Gender'].mode()[0], inplace=True)
test_data['Married'].fillna(test_data['Married'].mode()[0], inplace=True)
test_data['Dependents'].fillna(test_data['Dependents'].mode()[0], inplace=True)
test_data['Self_Employed'].fillna(test_data['Self_Employed'].mode()[0], inplace=True)

# Drop the Loan_ID column as it is not relevant for clustering
test_data_ids = test_data['Loan_ID']
test_data.drop('Loan_ID', axis=1, inplace=True)

# Encode categorical variables using the same encoders from training data
for column in test_data.select_dtypes(include=['object']).columns:
    test_data[column] = label_encoders[column].transform(test_data[column])

# Standardize the test data
scaled_test_data = scaler.transform(test_data)

# Predict loan approval status for test data
test_data['Loan_Status'] = test_data.apply(lambda x: predict_loan_approval(x.values), axis=1)

# Prepare the submission file
submission = pd.DataFrame({
    'Loan_ID': test_data_ids,
    'Loan_Status': label_encoders['Loan_Status'].transform(test_data['Loan_Status'])
})

submission.to_csv('submission.csv', index=False)

