<H3>Name:Dhivya Dharshini B</H3>
<H3>Reg No:212223240031</H3>
<H1 ALIGN =CENTER>EX. NO.1 Introduction to Kaggle and Data preprocessing</H1>

## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

## EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

**Kaggle :**
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

**Data Preprocessing:**

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

**Need of Data Preprocessing :**

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:
STEP 1:Importing the libraries<BR>
STEP 2:Importing the dataset<BR>
STEP 3:Taking care of missing data<BR>
STEP 4:Encoding categorical data<BR>
STEP 5:Normalizing the data<BR>
STEP 6:Splitting the data into test and train<BR>

##  PROGRAM:

```
import pandas as pd
import io
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

df = pd.read_csv("Churn_Modelling.csv")
df

df.isnull().sum()

df.fillna(0)
df.isnull().sum()

df.duplicated()

df['EstimatedSalary'].describe()

scaler = StandardScaler()
inc_cols = ['CreditScore', 'Tenure', 'Balance', 'EstimatedSalary']
scaled_values = scaler.fit_transform(df[inc_cols])
df[inc_cols] = pd.DataFrame(scaled_values, columns = inc_cols, index = df.index)
df

x = df.iloc[:, :-1]
y = df.iloc[:, -1]

print("X Values")
x

print("Y Values")
y

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)

print("X Training data")
x_train

print("X Testing data")
x_test

```
## OUTPUT:


# Read the dataset from drive


<img width="1040" height="332" alt="image" src="https://github.com/user-attachments/assets/95886d73-5ced-416e-9781-bec388c52d3c" />


# Finding Missing Values


<img width="148" height="413" alt="image" src="https://github.com/user-attachments/assets/da4cc688-b622-4835-8100-70bcc2d269bf" />


# Handling Missing values



<img width="131" height="423" alt="image" src="https://github.com/user-attachments/assets/a7a16c50-ec35-4847-b3ed-19b510026ca8" />



# Check for Duplicates


<img width="175" height="365" alt="image" src="https://github.com/user-attachments/assets/8d9b7ac4-cf02-474a-9405-2e896a1b0252" />

# Detect Outliers




<img width="192" height="254" alt="image" src="https://github.com/user-attachments/assets/76373161-9f91-4ad5-aee9-c80c671c8539" />



# Normalize the dataset




<img width="1039" height="367" alt="image" src="https://github.com/user-attachments/assets/8a952d37-3f80-4b58-8b03-7f4dcdc8f859" />




# Split the dataset into input and output




<img width="998" height="347" alt="image" src="https://github.com/user-attachments/assets/a05d50d7-c535-41d4-b291-1ea95b65256e" />




<img width="275" height="379" alt="image" src="https://github.com/user-attachments/assets/0a8d410b-45b0-4299-9bbb-0531c61d005d" />

# Print the training data and testing data




<img width="1003" height="351" alt="image" src="https://github.com/user-attachments/assets/c7697a45-94fd-4484-9b66-f908177a9601" />




<img width="967" height="322" alt="image" src="https://github.com/user-attachments/assets/fec07e45-9e7b-4247-a91e-08cf03d67ff2" />




## RESULT:
Thus, Implementation of Data Preprocessing is done in python  using a data set downloaded from Kaggle.


