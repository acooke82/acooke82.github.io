---
title: "Multi-Class Classifier"
excerpt: "A classifier that utilizes logistic regression to categorize TV/movie reviews into 3 categories. ![Internship Post Image](/images/classifier_image.png)"
collection: portfolio
---
_Tools & Technologies: Pandas, NumPy, scikit-learn_

### Summary

The goal of this project was to classify movie and TV show reviews into three different categories: (0) Not a movie or TV show review, (1) Positive review, or (2) Negative review. I decided to use logistic regression because it was the alogrithm that I had the most experience with, so I felt comfortable with being able to run and manipulate it with *scikit-learn*. 

A challenge I faced during my project was ensuring that the data was being preprocessed correctly and that any irrelevant texts were being filtered out of the data (gibberish, different languages, special characters). It would also be a great idea in the future to confirm that the labels of the training data are correct before using them to train our model. An ideal approach to do this task in the future could be to utilize TFIDFVectorizer, which would assist in weighting the frequency of the words as opposed to the binary method of the LogisticRegression classifier. This project was part of a course I completed during my MSc program, LING539.

### 1. Analyzing Data

For this project, I had been given two different .csv files, "train.csv" and "test.csv", to build my model. I used *pandas* to create DataFrames to read in both .csvs as utilizing a DataFrame simplified the process of pre-processing the data to create the necessary vectors. The train.csv contained 70,317 rows of data and three columns of "ID", "TEXT", and "LABEL". Each datapoint ID came pre-labeled with the "0, 1, 2" categories. The 0 category, "Not a Movie/TV Show Review", was the largest and contained 32,071 of IDs. The actual reviews, 1 (Postive Review) and 2 (Negative Review), contained 19,276 and 18,970 IDs, respectively. The "test.csv" contained 17,580 rows of data split into two columns, "ID" and "TEXT".

A quick review of the training data yielded the following observations. One, there is a significantly larger amount of "0" labels than there are "1" or "2" labels. Additionally, the "train.csv" dataset contained text that should not be used to train the model as it would impact the success of predictions on for "test.csv". There are some examples of data in a language different to that of the target language (English), such as ID 11352924827579021872 which contains "491. Из 9 фунтов муки испечено 16 белых хлебов. Сколько муки пошло на каждый?". For testing data, I also had a similar issue of data that will impact our model success. The "test.csv" contains texts in languages other than English, such as ID 6577057911690817390, which contains "對看講出無量法，老幼人人放喜懷". While reading in the .csvs using pandas, I also noticed a review that contained emojis, another type of special character that could impact model success. I opened and formatted the .csv files for analysis using the below code:

```python
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder

columns = ['ID', 'TEXT', 'LABEL']
data = pd.read_csv(r"C:\Users\Alex\Downloads\ling-539-sp-2024-class-competition_1\train.csv",header=0,names=columns)
data2 = pd.read_csv(r"C:\Users\Alex\Downloads\ling-539-sp-2024-class-competition_1\test.csv")

#to view the data format, uncomment the below:
#data.head()
#data2.head()
```

### 2. Training the Model

To train the model, I first needed to create vectors for the labels (0,1,2) and text from the training data. I made use of both *CountVectorizer* and *LabelEncoder* in order to convert the "TEXT" columns into feature vectors that could be recognized by the fit/predict aspects of the LogisticRegression classifier as well as to convert the "LABEL" columns into a label vector that would assign new values to our "test.csv" IDs. Please see below for how I created the vectors:

```python
#creating features from the training data text
cv = CountVectorizer()
x = cv.fit_transform(data['TEXT'].apply(lambda x: np.str_(x)))

#creating labels from the training data text
le = LabelEncoder()
y = le.fit_transform(data['LABEL'])
```

Once the vectors have been created, I could train my model using them and adjust the LogisticRegression parameters as necessary to create the final feature vector. This is a pretty quick process thanks to scikit-learn:

```python
#train the model
LR = LogisticRegression(max_iter=10000, random_state=42)
model = LR.fit(x,y)
```

### 3. Analyzing Test Data

Once the model has been fit succesfully, I could create a new feature vector for the 'TEXT' column that represents the test data:

```python
#creating features from the test data text
cv2 = CountVectorizer()
x2 = cv.transform(data2['TEXT'].apply(lambda x: np.str_(x)))
```

### 4. Predictions & Outputs 


```python
#get predictions
predictions = LR.predict(x2)

#to check the shape and length of the returned vector, uncomment the following:
#print(predictions)
#print(len(predictions))
```

Pandas DataFrames also simplified the process of outputting and saving my prediction results, as I could simply create a new .csv with headers and columns of my choice. 

```python
#creating .csv
csv = pd.DataFrame({"ID":list(data2["ID"]), "LABEL":list(predictions)})

#outputting it to my desktop for saving and sharing outside of Jupyter
csv.to_csv(r"C:\Users\Alex\Downloads\final3.csv",index=False)
```

### Results




