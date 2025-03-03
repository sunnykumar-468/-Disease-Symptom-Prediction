# -Disease-Symptom-Prediction
The Heart Attack Risk Prediction Dataset offers a comprehensive set of features to predict the likelihood of a heart attack based on various health and lifestyle factors. 
                                                        
# Import Important Library
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import pyplot
from pandas import read_csv
from pandas import set_option
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import Binarizer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from pandas import read_csv
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

print("All dependencies imported successfully")

# To Read Dataset
 df = pd.read_csv(r"/content/heart_attack_prediction_dataset.csv")
df

# Dataset Exploration
df.head()
df.tail()
df.info()
df.nunique()

for column in df.columns:
    unique_values = df[column].unique()
    print(f"Unique values in column '{column}': {unique_values}")

# Data Cleaning

# Convert binary columns (0/1) to "No"/"Yes"
for column in df.columns:
    if set(df[column].unique()) == {0, 1}:
        df[column] = df[column].replace({0: 'No', 1: 'Yes'})
df

for column in df.columns:
    unique_values = df[column].unique()
    print(f"Unique values in column '{column}': {unique_values}")

# Data Analysis
import seaborn as sns 

plt.figure(figsize=(8, 6))
sns.histplot(df['Age'], kde=False, bins=20, color='skyblue')
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

import seaborn as sns

plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='Sex', hue='Heart Attack Risk', palette='deep')
plt.title('Sex vs Heart Attack Risk')
plt.xlabel('Sex')
plt.ylabel('Count')
plt.legend(title='Heart Attack Risk', labels=['No', 'Yes'])
plt.show()

# Pie Chart Data
risk_by_sex = df.groupby(['Sex', 'Heart Attack Risk']).size().unstack(fill_value=0)
male_percentages = risk_by_sex.loc['Male'] / risk_by_sex.loc['Male'].sum() * 100
female_percentages = risk_by_sex.loc['Female'] / risk_by_sex.loc['Female'].sum() * 100

# Pie Chart for Male
plt.figure(figsize=(6, 6))
plt.pie(male_percentages, labels=['No Risk', 'At Risk'], autopct='%1.1f%%', startangle=90, colors=['#66b3ff', '#ff9999'])
plt.title('Heart Attack Risk Distribution (Male)')
plt.show()

# Pie Chart for Female
plt.figure(figsize=(6, 6))
plt.pie(female_percentages, labels=['No Risk', 'At Risk'], autopct='%1.1f%%', startangle=90, colors=['#99ff99', '#ffcc99'])
plt.title('Heart Attack Risk Distribution (Female)')
plt.show()

# Bar Plot for Age vs Heart Attack Risk
plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='Age', hue='Heart Attack Risk', palette='deep')
plt.title('Age vs Heart Attack Risk')
plt.xlabel('Age')
plt.ylabel('Count')
plt.legend(title='Heart Attack Risk', labels=['No', 'Yes'])
plt.show()

# Pie Chart Data
risk_by_age = df.groupby(['Age', 'Heart Attack Risk']).size().unstack(fill_value=0)
age_groups = risk_by_age.index

for age_group in age_groups:
    age_percentages = risk_by_age.loc[age_group] / risk_by_age.loc[age_group].sum() * 100
    
    # Pie Chart for each Age group
    plt.figure(figsize=(6, 6))
    plt.pie(
        age_percentages,
        labels=['No Risk', 'At Risk'],
        autopct='%1.1f%%',
        startangle=90,
        colors=['#66b3ff', '#ff9999']
    )
    plt.title(f'Heart Attack Risk Distribution ({age_group})')
    plt.show()
# Calculate percentages of 'Yes' for each age group
age_risk_percentages = risk_by_age.div(risk_by_age.sum(axis=1), axis=0) * 100

# Find the age groups with the lowest and highest percentage of "Yes"
lowest_risk_age = age_risk_percentages['Yes'].idxmin()
highest_risk_age = age_risk_percentages['Yes'].idxmax()

# Get the percentage for the lowest and highest risk age groups
lowest_risk_percentage = age_risk_percentages.loc[lowest_risk_age, 'Yes']
highest_risk_percentage = age_risk_percentages.loc[highest_risk_age, 'Yes']

# Plotting
plt.figure(figsize=(8, 6))
sns.barplot(x=age_risk_percentages.index, y=age_risk_percentages['Yes'], palette='coolwarm')

# Highlight the lowest and highest
plt.scatter(lowest_risk_age, lowest_risk_percentage, color='red', s=100, label=f"Lowest: {lowest_risk_age} ({lowest_risk_percentage:.2f}%)")
plt.scatter(highest_risk_age, highest_risk_percentage, color='green', s=100, label=f"Highest: {highest_risk_age} ({highest_risk_percentage:.2f}%)")

# Labels and title
plt.title('Heart Attack Risk Percentage by Age Group')
plt.xlabel('Age Group')
plt.ylabel('Percentage of Heart Attack Risk "Yes"')
plt.legend()
plt.show()

# Output the results
print(f"Age group with the lowest heart attack risk ('Yes'): {lowest_risk_age} ({lowest_risk_percentage:.2f}%)")
print(f"Age group with the highest heart attack risk ('Yes'): {highest_risk_age} ({highest_risk_percentage:.2f}%)")

# Calculate percentages of 'No' for each age group
age_no_percentages = risk_by_age['No'] / risk_by_age.sum(axis=1) * 100

# Find the age groups with the lowest and highest percentage of "No" for Heart Attack Risk
lowest_no_risk_age = age_no_percentages.idxmin()
highest_no_risk_age = age_no_percentages.idxmax()

# Plotting for "No" risk
plt.figure(figsize=(8, 6))
sns.barplot(x=age_no_percentages.index, y=age_no_percentages, palette='coolwarm')

# Highlight the lowest and highest for "No" risk
plt.scatter(lowest_no_risk_age, age_no_percentages.loc[lowest_no_risk_age], color='red', s=100, label=f"Lowest: {lowest_no_risk_age} ({age_no_percentages.loc[lowest_no_risk_age]:.2f}%)")
plt.scatter(highest_no_risk_age, age_no_percentages.loc[highest_no_risk_age], color='green', s=100, label=f"Highest: {highest_no_risk_age} ({age_no_percentages.loc[highest_no_risk_age]:.2f}%)")

# Labels and title
plt.title('Heart Attack Risk "No" Percentage by Age Group')
plt.xlabel('Age Group')
plt.ylabel('Percentage of Heart Attack Risk "No"')
plt.legend()
plt.show()

# Output the results
print(f"Age group with the lowest heart attack risk ('No'): {lowest_no_risk_age} ({age_no_percentages.loc[lowest_no_risk_age]:.2f}%)")
print(f"Age group with the highest heart attack risk ('No'): {highest_no_risk_age} ({age_no_percentages.loc[highest_no_risk_age]:.2f}%)")

# Calculate the new risk score (No - Yes) for each age group
age_risk_score = age_no_percentages - age_risk_percentages['Yes']

# Sort the age groups by the risk score (from highest to lowest)
ranked_age_risk_score = age_risk_score.sort_values(ascending=False)

# Plotting the ranked heart attack risk score by age group
plt.figure(figsize=(8, 6))
sns.barplot(x=ranked_age_risk_score.index, y=ranked_age_risk_score, palette='coolwarm')

# Highlight the most positive (lowest risk) and the most negative (highest risk) points
lowest_risk_age = ranked_age_risk_score.idxmax()
highest_risk_age = ranked_age_risk_score.idxmin()

plt.scatter(lowest_risk_age, ranked_age_risk_score.loc[lowest_risk_age], color='green', s=100, label=f"Lowest Risk: {lowest_risk_age} ({ranked_age_risk_score.loc[lowest_risk_age]:.2f})")
plt.scatter(highest_risk_age, ranked_age_risk_score.loc[highest_risk_age], color='red', s=100, label=f"Highest Risk: {highest_risk_age} ({ranked_age_risk_score.loc[highest_risk_age]:.2f})")

# Labels and title
plt.title('Heart Attack Risk Score (No - Yes) by Age Group')
plt.xlabel('Age Group')
plt.ylabel('Risk Score (No - Yes)')
plt.legend()
plt.show()

# Output the ranked results
print("Age groups ranked by Heart Attack Risk Score (No - Yes):")
print(ranked_age_risk_score)

# Cholestrol Distribution
plt.figure(figsize=(8, 6))
sns.histplot(df['Cholesterol'], kde=False, bins=20, color='lightblue')
plt.title('Cholesterol Distribution')
plt.xlabel('Cholesterol')
plt.ylabel('Frequency')
plt.show()

# Scatter plot for Cholesterol vs Heart Attack Risk
plt.figure(figsize=(8, 6))
sns.scatterplot(x='Cholesterol', y='Heart Attack Risk', data=df, hue='Heart Attack Risk', palette='coolwarm', s=100)

# Labels and title
plt.title('Cholesterol vs Heart Attack Risk')
plt.xlabel('Cholesterol Level')
plt.ylabel('Heart Attack Risk')
plt.legend(title='Heart Attack Risk')
plt.show()

#Boxplot
sns.boxplot(x='Heart Attack Risk', y='Cholesterol', data=df, palette='coolwarm')

#Histplot
sns.histplot(data=df, x='Cholesterol', hue='Heart Attack Risk', kde=True, palette='coolwarm', bins=30)

# BP Ratio Distribution
plt.figure(figsize=(8, 6))
sns.histplot(df['BP_Ratio'], kde=False, bins=20, color='lightblue')
plt.title('Blood Pressure Distribution')
plt.xlabel('BP Ratio')
plt.ylabel('Frequency')
plt.show()

# Systolic Distribution
plt.figure(figsize=(8, 6))
sns.histplot(df['Systolic'], kde=False, bins=20, color='lightblue')
plt.title('Systolic Distribution')
plt.xlabel('Systolic')
plt.ylabel('Frequency')
plt.show()

# Diastolic Distribution
plt.figure(figsize=(8, 6))
sns.histplot(df['Diastolic'], kde=False, bins=20, color='lightblue')
plt.title('Diastolic Distribution')
plt.xlabel('Diastolic')
plt.ylabel('Frequency')
plt.show()

# Plotting BP_Ratio against Heart Attack Risk
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='BP_Ratio', y='Heart Attack Risk', hue='Heart Attack Risk', palette='coolwarm', s=100, edgecolor='black')

# Labels and title
plt.title('BP Ratio vs Heart Attack Risk')
plt.xlabel('BP Ratio')
plt.ylabel('Heart Attack Risk')
plt.legend(title='Heart Attack Risk', labels=['No', 'Yes'])
plt.show()

# Boxplot of BP_Ratio by Heart Attack Risk
plt.figure(figsize=(8, 6))
sns.boxplot(data=df, x='Heart Attack Risk', y='BP_Ratio', palette='coolwarm')

# Labels and title
plt.title('BP Ratio by Heart Attack Risk')
plt.xlabel('Heart Attack Risk')
plt.ylabel('BP Ratio')
plt.show()

# FacetGrid to show BP_Ratio by Heart Attack Risk
g = sns.FacetGrid(df, col='Heart Attack Risk', palette='coolwarm', height=6, aspect=1.5)
g.map(sns.histplot, 'BP_Ratio', kde=True)

# Show plot
plt.show()

# Plotting a bar graph for Diabetes
plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='Diabetes', palette='deep')

# Labels and title
plt.title('Diabetes Distribution')
plt.xlabel('Diabetes')
plt.ylabel('Count')
plt.xticks([0, 1], ['No', 'Yes'])  
plt.show()

# Plotting Diabetes vs Heart Attack Risk
plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='Diabetes', hue='Heart Attack Risk', palette='deep')

# Labels and title
plt.title('Diabetes vs Heart Attack Risk')
plt.xlabel('Diabetes')
plt.ylabel('Count')
plt.xticks([0, 1], ['No', 'Yes'])  # Customize the x-axis labels if they are 0/1
plt.legend(title='Heart Attack Risk', labels=['No', 'Yes'])
plt.show()

# Count plot for Family History distribution
plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='Family History', palette='Set2')

# Labels and title
plt.title('Distribution of Family History')
plt.xlabel('Family History')
plt.ylabel('Count')
plt.xticks([0, 1], ['No', 'Yes'])  # Customize the x-axis labels if they are 0/1
plt.show()

# Count plot for Family History vs Heart Attack Risk
plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='Family History', hue='Heart Attack Risk', palette='deep')

# Labels and title
plt.title('Family History vs Heart Attack Risk')
plt.xlabel('Family History')
plt.ylabel('Count')
plt.xticks([0, 1], ['No', 'Yes'])  # Customize the x-axis labels if they are 0/1
plt.legend(title='Heart Attack Risk', labels=['No', 'Yes'])
plt.show()

# Count plot for Family History distribution
plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='Smoking', palette='Set2')

# Labels and title
plt.title('Distribution of Smoking')
plt.xlabel('Smoking')
plt.ylabel('Count')
plt.xticks([0, 1], ['No', 'Yes'])  # Customize the x-axis labels if they are 0/1
plt.show()

# Count plot for Family History vs Heart Attack Risk
plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='Smoking', hue='Heart Attack Risk', palette='deep')

# Labels and title
plt.title('Smoking vs Heart Attack Risk')
plt.xlabel('Smoking')
plt.ylabel('Count')
plt.xticks([0, 1], ['No', 'Yes'])  # Customize the x-axis labels if they are 0/1
plt.legend(title='Heart Attack Risk', labels=['No', 'Yes'])
plt.show()

# Count plot for Family History distribution
plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='Alcohol Consumption', palette='Set2')

# Labels and title
plt.title('Alcohol Consumption')
plt.xlabel('Alcohol Consumption')
plt.ylabel('Count')
plt.xticks([0, 1], ['No', 'Yes'])  
plt.show()

# Count plot for Family History vs Heart Attack Risk
plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='Alcohol Consumption', hue='Heart Attack Risk', palette='deep')

# Labels and title
plt.title('Alcohol Consumption vs Heart Attack Risk')
plt.xlabel('Alcohol Consumption')
plt.ylabel('Count')
plt.xticks([0, 1], ['No', 'Yes'])  
plt.legend(title='Heart Attack Risk', labels=['No', 'Yes'])
plt.show()

# Count plot for Family History distribution
plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='Diet', palette='Set2')

# Labels and title
plt.title('Diet')
plt.xlabel('Diet')
plt.ylabel('Count')
plt.xticks([0, 1, 2], ['Unhealthy', 'Average', 'Healthy' ])  
plt.show()

# Count plot for Family History vs Heart Attack Risk
plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='Diet', hue='Heart Attack Risk', palette='deep')

# Labels and title
plt.title('Diet vs Heart Attack Risk')
plt.xlabel('Diet')
plt.ylabel('Count')
plt.xticks([0, 1, 2], ['Unhealthy', 'Average', 'Healthy' ])  
plt.legend(title='Heart Attack Risk', labels=['No', 'Yes'])
plt.show()

# Count plot for Family History vs Heart Attack Risk
plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='Previous Heart Problems', hue='Heart Attack Risk', palette='deep')

# Labels and title
plt.title('Previous Heart Problems vs Heart Attack Risk')
plt.xlabel('Previous Heart Problemsn')
plt.ylabel('Count')
plt.xticks([0, 1], ['No', 'Yes'])  
plt.legend(title='Heart Attack Risk', labels=['No', 'Yes'])
plt.show()

plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='Medication Use', palette='Set2')

# Labels and title
plt.title('Medication Use')
plt.xlabel('Medication Use')
plt.ylabel('Count')
plt.xticks([0, 1], ['No', 'Yes'])  
plt.show()

# Count plot for Family History vs Heart Attack Risk
plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='Medication Use', hue='Heart Attack Risk', palette='deep')

# Labels and title
plt.title('Medication Use vs Heart Attack Risk')
plt.xlabel('Medication Use')
plt.ylabel('Count')
plt.xticks([0, 1], ['No', 'Yes'])  
plt.legend(title='Heart Attack Risk', labels=['No', 'Yes'])
plt.show()

plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='Stress Level', palette='Set2')

# Labels and title
plt.title('Stress Level')
plt.xlabel('Stress Level')
plt.ylabel('Count')
plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9 ], ['1','2','3','4','5','6','7','8','9','10'])  
plt.show()

plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='Stress Level', hue='Heart Attack Risk', palette='deep')

# Labels and title
plt.title('Stress Level vs Heart Attack Risk')
plt.xlabel('Stress Level')
plt.ylabel('Count')
plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9 ], ['1','2','3','4','5','6','7','8','9','10'])  
plt.legend(title='Heart Attack Risk', labels=['No', 'Yes'])
plt.show()

plt.figure(figsize=(8, 6))
sns.histplot(df['BMI'], kde=False, bins=20, color='lightblue')
plt.title('BMI Distribution')
plt.xlabel('BMI')
plt.ylabel('Frequency')
plt.show()

# FacetGrid to show BP_Ratio by Heart Attack Risk
g = sns.FacetGrid(df, col='Heart Attack Risk', palette='coolwarm', height=6, aspect=1.5)
g.map(sns.histplot, 'BMI', kde=True)

# Show plot
plt.show()

# Plotting a bar graph for Diabetes
plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='Diabetes', palette='deep')

# Labels and title
plt.title('Diabetes Distribution')
plt.xlabel('Diabetes')
plt.ylabel('Count')
plt.xticks([0, 1], ['No', 'Yes'])  
plt.show()

# Plotting Diabetes vs Heart Attack Risk
plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='Diabetes', hue='Heart Attack Risk', palette='deep')

# Labels and title
plt.title('Diabetes vs Heart Attack Risk')
plt.xlabel('Diabetes')
plt.ylabel('Count')
plt.xticks([0, 1], ['No', 'Yes'])  # Customize the x-axis labels if they are 0/1
plt.legend(title='Heart Attack Risk', labels=['No', 'Yes'])
plt.show()

# Count plot for Family History distribution
plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='Family History', palette='Set2')

# Labels and title
plt.title('Distribution of Family History')
plt.xlabel('Family History')
plt.ylabel('Count')
plt.xticks([0, 1], ['No', 'Yes'])  # Customize the x-axis labels if they are 0/1
plt.show()

# Count plot for Family History vs Heart Attack Risk
plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='Family History', hue='Heart Attack Risk', palette='deep')

# Labels and title
plt.title('Family History vs Heart Attack Risk')
plt.xlabel('Family History')
plt.ylabel('Count')
plt.xticks([0, 1], ['No', 'Yes'])  # Customize the x-axis labels if they are 0/1
plt.legend(title='Heart Attack Risk', labels=['No', 'Yes'])
plt.show()

# Count plot for Family History distribution
plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='Smoking', palette='Set2')

# Labels and title
plt.title('Distribution of Smoking')
plt.xlabel('Smoking')
plt.ylabel('Count')
plt.xticks([0, 1], ['No', 'Yes'])  # Customize the x-axis labels if they are 0/1
plt.show()

# Count plot for Family History vs Heart Attack Risk
plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='Smoking', hue='Heart Attack Risk', palette='deep')

# Labels and title
plt.title('Smoking vs Heart Attack Risk')
plt.xlabel('Smoking')
plt.ylabel('Count')
plt.xticks([0, 1], ['No', 'Yes'])  # Customize the x-axis labels if they are 0/1
plt.legend(title='Heart Attack Risk', labels=['No', 'Yes'])
plt.show()

# Count plot for Family History distribution
plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='Alcohol Consumption', palette='Set2')

# Labels and title
plt.title('Alcohol Consumption')
plt.xlabel('Alcohol Consumption')
plt.ylabel('Count')
plt.xticks([0, 1], ['No', 'Yes'])  
plt.show()

# Count plot for Family History vs Heart Attack Risk
plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='Alcohol Consumption', hue='Heart Attack Risk', palette='deep')

# Labels and title
plt.title('Alcohol Consumption vs Heart Attack Risk')
plt.xlabel('Alcohol Consumption')
plt.ylabel('Count')
plt.xticks([0, 1], ['No', 'Yes'])  
plt.legend(title='Heart Attack Risk', labels=['No', 'Yes'])
plt.show()

# Count plot for Family History distribution
plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='Diet', palette='Set2')

# Labels and title
plt.title('Diet')
plt.xlabel('Diet')
plt.ylabel('Count')
plt.xticks([0, 1, 2], ['Unhealthy', 'Average', 'Healthy' ])  
plt.show()

# Count plot for Family History vs Heart Attack Risk
plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='Diet', hue='Heart Attack Risk', palette='deep')

# Labels and title
plt.title('Diet vs Heart Attack Risk')
plt.xlabel('Diet')
plt.ylabel('Count')
plt.xticks([0, 1, 2], ['Unhealthy', 'Average', 'Healthy' ])  
plt.legend(title='Heart Attack Risk', labels=['No', 'Yes'])
plt.show()

# Count plot for Family History vs Heart Attack Risk
plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='Previous Heart Problems', hue='Heart Attack Risk', palette='deep')

# Labels and title
plt.title('Previous Heart Problems vs Heart Attack Risk')
plt.xlabel('Previous Heart Problemsn')
plt.ylabel('Count')
plt.xticks([0, 1], ['No', 'Yes'])  
plt.legend(title='Heart Attack Risk', labels=['No', 'Yes'])
plt.show()

plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='Medication Use', palette='Set2')

# Labels and title
plt.title('Medication Use')
plt.xlabel('Medication Use')
plt.ylabel('Count')
plt.xticks([0, 1], ['No', 'Yes'])  
plt.show()

# Count plot for Family History vs Heart Attack Risk
plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='Medication Use', hue='Heart Attack Risk', palette='deep')

# Labels and title
plt.title('Medication Use vs Heart Attack Risk')
plt.xlabel('Medication Use')
plt.ylabel('Count')
plt.xticks([0, 1], ['No', 'Yes'])  
plt.legend(title='Heart Attack Risk', labels=['No', 'Yes'])
plt.show()

plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='Stress Level', palette='Set2')

# Labels and title
plt.title('Stress Level')
plt.xlabel('Stress Level')
plt.ylabel('Count')
plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9 ], ['1','2','3','4','5','6','7','8','9','10'])  
plt.show()

plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='Stress Level', hue='Heart Attack Risk', palette='deep')

# Labels and title
plt.title('Stress Level vs Heart Attack Risk')
plt.xlabel('Stress Level')
plt.ylabel('Count')
plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9 ], ['1','2','3','4','5','6','7','8','9','10'])  
plt.legend(title='Heart Attack Risk', labels=['No', 'Yes'])
plt.show()

plt.figure(figsize=(8, 6))
sns.histplot(df['BMI'], kde=False, bins=20, color='lightblue')
plt.title('BMI Distribution')
plt.xlabel('BMI')
plt.ylabel('Frequency')
plt.show()

# FacetGrid to show BP_Ratio by Heart Attack Risk
g = sns.FacetGrid(df, col='Heart Attack Risk', palette='coolwarm', height=6, aspect=1.5)
g.map(sns.histplot, 'BMI', kde=True)

# Show plot
plt.show()

# Plotting Country vs Heart Attack Risk
plt.figure(figsize=(12, 8))
sns.countplot(data=df, x='Country', hue='Heart Attack Risk', palette='Set2')

# Labels and title
plt.title('Country vs Heart Attack Risk')
plt.xlabel('Country')
plt.ylabel('Count')
plt.xticks(rotation=90)  # Rotate country labels to fit them
plt.legend(title='Heart Attack Risk', labels=['No', 'Yes'])
plt.show()

# Grouping data by 'Country' and 'Heart Attack Risk' to get the count
risk_by_country = df.groupby(['Country', 'Heart Attack Risk']).size().unstack(fill_value=0)

# Calculate percentages for each country
risk_percentages_by_country = risk_by_country.div(risk_by_country.sum(axis=1), axis=0) * 100

# Plotting pie charts for each country
for country in risk_percentages_by_country.index:
    plt.figure(figsize=(6, 6))
    plt.pie(risk_percentages_by_country.loc[country], labels=['No Risk', 'At Risk'], autopct='%1.1f%%', startangle=90, colors=['#66b3ff', '#ff9999'])
    plt.title(f'Heart Attack Risk Distribution in {country}')
    plt.show()

# Plotting Continent vs Heart Attack Risk
plt.figure(figsize=(12, 8))
sns.countplot(data=df, x='Continent', hue='Heart Attack Risk', palette='Set2')

# Labels and title
plt.title('Continent vs Heart Attack Risk')
plt.xlabel('Continent')
plt.ylabel('Continent')
plt.xticks(rotation=90)  # Rotate country labels to fit them
plt.legend(title='Heart Attack Risk', labels=['No', 'Yes'])
plt.show()

# Plotting Hemisphere vs Heart Attack Risk
plt.figure(figsize=(12, 8))
sns.countplot(data=df, x='Hemisphere', hue='Heart Attack Risk', palette='Set2')

# Labels and title
plt.title('Hemisphere vs Heart Attack Risk')
plt.xlabel('Hemisphere')
plt.ylabel('Hemisphere')
plt.xticks(rotation=90)  # Rotate country labels to fit them
plt.legend(title='Heart Attack Risk', labels=['No', 'Yes'])
plt.show()

# To Encode Categorical Data
df.info()
for column in df.columns:
    unique_values = df[column].unique()
    print(f"Unique values in column '{column}': {unique_values}")

# Label Encoding
# Loop through all columns and apply encoding for 'Yes'/'No' values
for column in df.columns:
    if df[column].dtype == 'object' and df[column].isin(['Yes', 'No']).all():
        df[column] = df[column].map({'No': 0, 'Yes': 1})

# Verify the encoding
df

for column in df.columns:
    unique_values = df[column].unique()
    print(f"Unique values in column '{column}': {unique_values}")

# Encoding 'Sex' column: Female = 0, Male = 1
df['Sex'] = df['Sex'].map({'Female': 0, 'Male': 1})

# Encoding 'Diet' column: Unhealthy = 0, Average = 1, Healthy = 2
df['Diet'] = df['Diet'].map({'Unhealthy': 0, 'Average': 1, 'Healthy': 2})

# Verify the encoding
print(df[['Sex', 'Diet']].head())

# One Hot Encoding
# One-Hot Encoding for 'Continent' and 'Hemisphere'
df = pd.get_dummies(df, columns=['Continent', 'Hemisphere'], drop_first=True)

# Verify the changes
print(df.head())

df
# Reorder columns: move 'Heart Attack Risk' to the most right
cols = [col for col in df.columns if col != 'Heart Attack Risk']  # All columns except 'Heart Attack Risk'
df = df[cols + ['Heart Attack Risk']]  # Append 'Heart Attack Risk' to the rightmost position

# Verify the changes
df

# Data Cleaning
df = df.drop(columns="Country")
df

# To Verify Data Types
df.info()

# Feature Distribution
df_features = df.iloc[:, :-1]

# Visualize the distribution of all features except the target
def plot_distributions(df, cols_per_row=3):
    num_cols = len(df.columns)
    num_rows = (num_cols // cols_per_row) + int(num_cols % cols_per_row > 0)

    plt.figure(figsize=(cols_per_row * 6, num_rows * 5))
    for i, column in enumerate(df.columns, 1):
        plt.subplot(num_rows, cols_per_row, i)
        sns.histplot(df[column], kde=True, bins=30, color='blue', edgecolor='black')
        plt.title(f"Distribution of {column}", fontsize=14)  # Increased title font size
        plt.xlabel(column, fontsize=12)  # Increased x-label font size
        plt.ylabel("Frequency", fontsize=12)  # Increased y-label font size
        plt.tight_layout()
    plt.show()

# Call the function to plot distributions
plot_distributions(df_features, cols_per_row=3)

# For Discriptive Statistical Analysis
df.describe()

# To Measure Central Tendency
# Select column
columns_to_analyze = df.columns[0:29]

# Calculate mean, median, and mode for each column
mean_values = df[columns_to_analyze].mean()
median_values = df[columns_to_analyze].median()
mode_values = df[columns_to_analyze].mode().iloc[0]

# Display the results
print("Mean values for features:")
print(mean_values)
print("\nMedian values for features:")
print(median_values)
print("\nMode values for features:")
print(mode_values)

# To Measure Variability
# Select column
columns_to_analyze = df.columns[0:29]

# Calculate standard deviation, variance, and range for each numerical column
std_values = df[columns_to_analyze].std()
variance_values = df[columns_to_analyze].var()
range_values = df[columns_to_analyze].max() - df[columns_to_analyze].min()

# Display the results
print("Standard Deviation values for numerical features:")
print(std_values)
print("\nVariance values for numerical features:")
print(variance_values)
print("\nRange values for numerical features:")
print(range_values)

# To Check Outliers
# Plot boxplots for numerical columns
plt.figure(figsize=(12, 8))
sns.boxplot(data=df)
plt.xticks(rotation=90)
plt.title('Box Plots for Numerical Features')
plt.show()

# Verifying Outliers By IQR
# Loop through all columns except target label
for column in df.columns:
    if column != 'diagnosis':
        # Calculate Q1, Q3, and IQR for each column
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1

        # Define the outlier thresholds
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Check for outliers
        outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]

        # Print if no outliers are found for the column
        if outliers.empty:
            print(f"No outliers detected in {column}")
        else:
            print(f"Outliers detected in {column}:")
            print(outliers[[column]])

# Verifying Outliers By Z-Scoring
from scipy.stats import zscore

# Select only the numeric columns for Z-score calculation
numeric_cols = df.select_dtypes(include=[np.number])

# Calculate Z-scores
z_scores = zscore(numeric_cols)

# Convert Z-scores into a DataFrame
z_scores_df = pd.DataFrame(z_scores, columns=numeric_cols.columns)

# Define the threshold for outliers (common value is 3 or -3)
outliers = (z_scores_df.abs() > 3)

# Show rows with outliers
outlier_rows = df[outliers.any(axis=1)]

# Output the outlier rows
print(outlier_rows)

# Measurement of Skewness
# Summary statistics for all numeric columns
print("Summary Statistics for Numeric Columns:")
print(df.describe())

# Check skewness for all numeric columns
from scipy.stats import skew

print("\nSkewness for All Numeric Columns:")
skewness = df.select_dtypes(include=['number']).apply(skew)
print(skewness)

# Density Plot
# Select numeric columns
numeric_columns = df.select_dtypes(include=['number']).columns

# Generate density plots for all numeric columns
plt.figure(figsize=(10, len(numeric_columns) * 5))  # Adjust figure size based on the number of columns
for i, column in enumerate(numeric_columns, 1):
    plt.subplot(len(numeric_columns), 1, i)
    sns.kdeplot(df[column], shade=True, color='blue')
    plt.title(f'Density Plot of {column}')
    plt.xlabel(column)
    plt.ylabel('Density')

plt.tight_layout()
plt.show()

# To Find Correlation
df.corr()

# Correlation Heatmap
# Compute the correlation matrix
correlation_matrix = df.corr()

# Create a heatmap to visualize the correlation matrix
plt.figure(figsize=(12, 8))  # Adjust figure size
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)

# Customize the plot
plt.title('Correlation Heatmap', fontsize=16)
plt.show()

# Feature Correlation by Ranking
correlation_with_success = correlation_matrix['Heart Attack Risk']

# Sort the correlation values in descending order
correlation_ranked = correlation_with_success.sort_values(ascending=False)

# Print the ranked correlation values
print(correlation_ranked)

# Feature Relationship
# Pairplot to visualize relationships
sns.pairplot(df, diag_kind="kde")
plt.suptitle("Pairwise Relationships Between Attributes", y=1.02)
plt.show()

# Feature Importance
from sklearn.ensemble import RandomForestRegressor

# Prepare data
X = df.drop(columns=[df.columns[-1]])  # Drop the last column (target)
y = df[df.columns[-1]]  # Select the last column (target

# Train a Random Forest Regressor model
rf = RandomForestRegressor()
rf.fit(X, y)

# Get feature importances
importances = rf.feature_importances_

# Create a DataFrame for feature importances
feature_importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

# Plot the feature importance
plt.figure(figsize=(10, 6))
sns.barplot(data=feature_importance_df, x='Importance', y='Feature')
plt.title('Feature Importance for Continuous Target Prediction')
plt.show()

# Get the top 5 features
df_top5_importance = feature_importance_df.head(5)

# Create a new DataFrame with only the top 5 features and the target
df_top5_importance = X[df_top5_importance['Feature']].copy()
df_top5_importance['Heart Attack Risk'] = y  # Add the target column

df_top5_importance

# Feature Selection RFE
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression 

# Create the model
model = LogisticRegression() 

# Create the RFE model and select the top 5 features 
rfe = RFE(estimator=model, n_features_to_select=5)
rfe.fit(X, y)

# Print the features selected by RFE
selected_features = X.columns[rfe.support_]
print(f"Selected Features: {selected_features}")

# Print the ranking of all features
print(f"Feature Ranking: {rfe.ranking_}")

# Retain only the selected features in the DataFrame
df_rfe = df[selected_features]

# Optionally, you can add the target column ('Air Quality') back to df_selected
df_rfe['Heart Attack Risk'] = y

df_rfe

# Feature Scalling
from sklearn.preprocessing import MinMaxScaler

# Convert DataFrame to numpy array
array = df.values

# Separate array into input (X) and output (Y) components
X = array[:, :-1]  # Input features (all rows, all columns except the last one)
Y = array[:, -1]   # Output target (all rows, only the last column)

# Initialize MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))

# Rescale the input features
rescaledX = scaler.fit_transform(X)

# Print first 5 rescaled inputs
print(rescaledX[0:5, :])

# Feature Importance Data
from sklearn.preprocessing import MinMaxScaler

# Convert DataFrame to numpy array
array = df_top5_importance.values

# Separate array into input (X) and output (Y) components
X = array[:, :-1]  # Input features (all rows, all columns except the last one)
Y = array[:, -1]   # Output target (all rows, only the last column)

# Initialize MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))

# Rescale the input features
rescaledX_top5 = scaler.fit_transform(X)

# Print first 5 rescaled inputs
print(rescaledX_top5[0:5, :])

# Min-Max RFE
from sklearn.preprocessing import MinMaxScaler

# Convert DataFrame to numpy array
array = df_rfe.values

# Separate array into input (X) and output (Y) components
X = array[:, :-1]  # Input features (all rows, all columns except the last one)
Y = array[:, -1]   # Output target (all rows, only the last column)

# Initialize MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))

# Rescale the input features
rescaledX_rfe = scaler.fit_transform(X)

# Print first 5 rescaled inputs
print(rescaledX_rfe[0:5, :])

# Data Preparing
import pandas as pd
from sklearn.model_selection import train_test_split

array = df.values

# Extract X (features) and y (target)
X = rescaledX
y = Y

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Feature Importance Data
import pandas as pd
from sklearn.model_selection import train_test_split

array = df_top5_importance.values

# Extract X (features) and y (target)
X_top5 = rescaledX_top5
y_top5 = Y

# Split data into training and testing sets
X_train_top5, X_test_top5, y_train_top5, y_test_top5 = train_test_split(X_top5, y_top5, test_size=0.3, random_state=42)

# RFE Data 
import pandas as pd
from sklearn.model_selection import train_test_split

array = df_rfe.values

# Extract X (features) and y (target)
X_rfe = rescaledX_rfe
y_rfe = Y

# Split data into training and testing sets
X_train_rfe, X_test_rfe, y_train_rfe, y_test_rfe = train_test_split(X_rfe, y_rfe, test_size=0.3, random_state=42)

# Checking for Data Imbalance 
# Convert y_train to pandas Series if it is a NumPy array
y_train_series = pd.Series(y_train)

# Calculate the class distribution in the training data
class_dist_train = y_train_series.value_counts()
print("Class distribution in training data:")
print(class_dist_train)
print()

# Calculate the percentage for each class in the training data
total_samples_train = class_dist_train.sum()
percentages_train = (class_dist_train / total_samples_train) * 100
print("Class Percentages in Training Data:")
print(percentages_train)

# Convert y_train to pandas Series if it is a NumPy array
y_train_series = pd.Series(y_train_top5)

# Calculate the class distribution in the training data
class_dist_train = y_train_series.value_counts()
print("Class distribution in training data:")
print(class_dist_train)
print()

# Calculate the percentage for each class in the training data
total_samples_train = class_dist_train.sum()
percentages_train = (class_dist_train / total_samples_train) * 100
print("Class Percentages in Training Data:")
print(percentages_train)

# Convert y_train to pandas Series if it is a NumPy array
y_train_series = pd.Series(y_train_rfe)

# Calculate the class distribution in the training data
class_dist_train = y_train_series.value_counts()
print("Class distribution in training data:")
print(class_dist_train)
print()

# Calculate the percentage for each class in the training data
total_samples_train = class_dist_train.sum()
percentages_train = (class_dist_train / total_samples_train) * 100
print("Class Percentages in Training Data:")
print(percentages_train)

#SMOTE Oversampling for Minority Class
from imblearn.over_sampling import SMOTE

# Convert y_train_top5 and y_train_rfe to pandas Series for easier manipulation
y_train = pd.Series(y_train)
y_train_top5 = pd.Series(y_train_top5)
y_train_rfe = pd.Series(y_train_rfe)

# Display class distribution before oversampling
print('Before Oversampling for X_train:')
print(y_train.value_counts())

print('Before Oversampling for X_train_top5:')
print(y_train_top5.value_counts())

print('Before Oversampling for X_train_rfe:')
print(y_train_rfe.value_counts())

# Apply SMOTE for oversampling on the full training set
smote = SMOTE(random_state=42)

# Oversample X_train, y_train
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Oversample X_train_top5, y_train_top5
X_train_top5_smote, y_train_top5_smote = smote.fit_resample(X_train_top5, y_train_top5)

# Oversample X_train_rfe, y_train_rfe
X_train_rfe_smote, y_train_rfe_smote = smote.fit_resample(X_train_rfe, y_train_rfe)

# Convert y_train_smote, y_train_top5_smote, and y_train_rfe_smote back to pandas Series
y_train_smote = pd.Series(y_train_smote)
y_train_top5_smote = pd.Series(y_train_top5_smote)
y_train_rfe_smote = pd.Series(y_train_rfe_smote)

# Display class distribution after oversampling
print('After Oversampling for X_train:')
print(y_train_smote.value_counts())

print('After Oversampling for X_train_top5:')
print(y_train_top5_smote.value_counts())

print('After Oversampling for X_train_rfe:')
print(y_train_rfe_smote.value_counts())

# Applying Supervised Learning
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score

# Prepare models
models = [
    ('Logistic Regression', LogisticRegression(solver='liblinear')),
    ('KNN', KNeighborsClassifier()),
    ('Decision Tree', DecisionTreeClassifier()),
    ('Naive Bayes', GaussianNB()),
    ('Support Vector Machine', SVC()),
    ('GBM', GradientBoostingClassifier()),
    ('XGBoost', xgb.XGBClassifier(eval_metric='mlogloss')),
    ('MLP', MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500))
]

# Prepare lists to store results and names
results = []
names = []

# Evaluate each model in turn
for name, model in models:
    # Train the model using training data (SMOTE for handling imbalance)
    model.fit(X_train_smote, y_train_smote)

    # Predict on the test set
    y_pred = model.predict(X_test)

    # Accuracy score
    accuracy = model.score(X_test, y_test)
    results.append(accuracy)
    names.append(name)
    print(f"{name}: Accuracy: {accuracy:.3f}")
    
    # Classification report
    print(f"Classification Report for {name}:\n", classification_report(y_test, y_pred))
    print()

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", 
                xticklabels=['Healthy', 'Heart Attack'], 
                yticklabels=['Healthy', 'Heart Attack'])
    plt.title(f'Confusion Matrix for {name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

# Bar plot for algorithm comparison (accuracies of models)
plt.figure(figsize=(10, 6))
sns.barplot(x=names, y=results, palette='viridis')
plt.title('Algorithm Comparison: Model Accuracy')
plt.xlabel('Models')
plt.ylabel('Accuracy')
plt.ylim(0, 1)  # Limiting y-axis from 0 to 1 for accuracy percentage
plt.xticks(rotation=45)  # Rotate model names for better readability
plt.show()

# Feature Importance Data
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score

# Prepare models
models = [
    ('Logistic Regression', LogisticRegression(solver='liblinear')),
    ('KNN', KNeighborsClassifier()),
    ('Decision Tree', DecisionTreeClassifier()),
    ('Naive Bayes', GaussianNB()),
    ('Support Vector Machine', SVC()),
    ('GBM', GradientBoostingClassifier()),
    ('XGBoost', xgb.XGBClassifier(eval_metric='mlogloss')),
    ('MLP', MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500))
]

# Prepare lists to store results and names
results = []
names = []

# Evaluate each model in turn
for name, model in models:
    # Train the model using training data (SMOTE for handling imbalance)
    model.fit(X_train_top5_smote, y_train_top5_smote)

    # Predict on the test set
    y_pred = model.predict(X_test_top5)

    # Accuracy score
    accuracy = model.score(X_test_top5, y_test_top5)
    results.append(accuracy)
    names.append(name)
    print(f"{name}: Accuracy: {accuracy:.3f}")
    
    # Classification report
    print(f"Classification Report for {name}:\n", classification_report(y_test, y_pred))
    print()

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", 
                xticklabels=['Healthy', 'Heart Attack'], 
                yticklabels=['Healthy', 'Heart Attack'])
    plt.title(f'Confusion Matrix for {name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

# Bar plot for algorithm comparison (accuracies of models)
plt.figure(figsize=(10, 6))
sns.barplot(x=names, y=results, palette='viridis')
plt.title('Algorithm Comparison: Model Accuracy')
plt.xlabel('Models')
plt.ylabel('Accuracy')
plt.ylim(0, 1)  # Limiting y-axis from 0 to 1 for accuracy percentage
plt.xticks(rotation=45)  # Rotate model names for better readability
plt.show()

#RFE Data
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score

# Prepare models
models = [
    ('Logistic Regression', LogisticRegression(solver='liblinear')),
    ('KNN', KNeighborsClassifier()),
    ('Decision Tree', DecisionTreeClassifier()),
    ('Naive Bayes', GaussianNB()),
    ('Support Vector Machine', SVC()),
    ('GBM', GradientBoostingClassifier()),
    ('XGBoost', xgb.XGBClassifier(eval_metric='mlogloss')),
    ('MLP', MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500))
]

# Prepare lists to store results and names
results = []
names = []

# Evaluate each model in turn
for name, model in models:
    # Train the model using training data (SMOTE for handling imbalance)
    model.fit(X_train_rfe, y_train_rfe)

    # Predict on the test set
    y_pred = model.predict(X_test_rfe)

    # Accuracy score
    accuracy = model.score(X_test_rfe, y_test_rfe)
    results.append(accuracy)
    names.append(name)
    print(f"{name}: Accuracy: {accuracy:.3f}")
    
    # Classification report
    print(f"Classification Report for {name}:\n", classification_report(y_test, y_pred))
    print()

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", 
                xticklabels=['Healthy', 'Heart Attack'], 
                yticklabels=['Healthy', 'Heart Attack'])
    plt.title(f'Confusion Matrix for {name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

# Bar plot for algorithm comparison (accuracies of models)
plt.figure(figsize=(10, 6))
sns.barplot(x=names, y=results, palette='viridis')
plt.title('Algorithm Comparison: Model Accuracy')
plt.xlabel('Models')
plt.ylabel('Accuracy')
plt.ylim(0, 1)  # Limiting y-axis from 0 to 1 for accuracy percentage
plt.xticks(rotation=45)  # Rotate model names for better readability
plt.show()

# Prediction Output
# Dictionary to store predictions for each model
test_predictions = {}

# Loop through each model to train and make predictions
for name, model in models:
    # Train the model
    model.fit(X_train_smote, y_train_smote)

    # Predict on the test set
    y_pred = model.predict(X_test)

    # Store predictions in the dictionary
    test_predictions[name] = y_pred

# Convert the predictions dictionary into a DataFrame
predictions_df = pd.DataFrame(test_predictions)

# Add the actual values for comparison
predictions_df['Actual'] = y_test

# Reorder the columns to put the 'Actual' value in the first column
predictions_df = predictions_df[['Actual'] + [col for col in predictions_df.columns if col != 'Actual']]

# Mapping to rename class labels
label_mapping = {0: 'Healthy', 1: 'Heart Attack'}

# Rename the values in the DataFrame using the mapping
predictions_df = predictions_df.replace(label_mapping)

# Display the predictions DataFrame
print("Table of Predictions:")
predictions_df

# Features Importance Data
# Dictionary to store predictions for each model
test_predictions = {}

# Loop through each model to train and make predictions
for name, model in models:
    # Train the model
    model.fit(X_train_smote, y_train_smote)

    # Predict on the test set
    y_pred = model.predict(X_test)

    # Store predictions in the dictionary
    test_predictions[name] = y_pred

# Convert the predictions dictionary into a DataFrame
predictions_df = pd.DataFrame(test_predictions)

# Add the actual values for comparison
predictions_df['Actual'] = y_test

# Reorder the columns to put the 'Actual' value in the first column
predictions_df = predictions_df[['Actual'] + [col for col in predictions_df.columns if col != 'Actual']]

# Mapping to rename class labels
label_mapping = {0: 'Healthy', 1: 'Heart Attack'}

# Rename the values in the DataFrame using the mapping
predictions_df = predictions_df.replace(label_mapping)

# Display the predictions DataFrame
print("Table of Predictions:")
predictions_df

# RFE data
# Dictionary to store predictions for each model
test_predictions = {}

# Loop through each model to train and make predictions
for name, model in models:
    # Train the model
    model.fit(X_train_rfe, y_train_rfe)

    # Predict on the test set
    y_pred = model.predict(X_test_rfe)

    # Store predictions in the dictionary
    test_predictions[name] = y_pred

# Convert the predictions dictionary into a DataFrame
predictions_df = pd.DataFrame(test_predictions)

# Add the actual values for comparison
predictions_df['Actual'] = y_test

# Reorder the columns to put the 'Actual' value in the first column
predictions_df = predictions_df[['Actual'] + [col for col in predictions_df.columns if col != 'Actual']]

# Mapping to rename class labels
label_mapping = {0: 'Healthy', 1: 'Heart Attack'}

# Rename the values in the DataFrame using the mapping
predictions_df = predictions_df.replace(label_mapping)

# Display the predictions DataFrame
print("Table of Predictions:")
predictions_df























