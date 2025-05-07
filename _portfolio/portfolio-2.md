---
title: "Multi-Class Classifier"
excerpt: "A classifier that utilizes logistic regression to categorize TV/movie reviews into 3 categories. ![Internship Post Image](/images/classifier_image.png)"
collection: portfolio
---
_Tools & Technologies: Pandas, NumPy, scikit-learn_

### Summary

The goal of this project was to build a model that could classify text passages into three different categories: (0) Not a movie or TV show review, (1) Positive review, or (2) Negative review. I decided to use logistic regression to build my classifier model because it was the algorithm that I had the most experience with, so I felt comfortable with being able to run and manipulate it by using *scikit-learn*. Once I had built my model, I judged its performance by calculating the F1 score. An F1 score is a good indicator of a model's predictive performance as it combines the harmonic means of precision and recall scores, ensuring a balanced score.  

This project was part of a [course](https://catalog.arizona.edu/courses/0199291) on statistical natural language processing I completed during the UA's MSc program. To view the code used for this project, please go [here](https://github.com/acooke82/multiclass_classification_model).

### 1. Exploratory Data Analysis

For this project, I had been given two different .csv files, "train.csv" and "test.csv", to build my model. I used *pandas* to create DataFrames to read in both .csvs as utilizing a DataFrame simplified the process of pre-processing the data to create the necessary vectors. The train.csv contained 70,317 rows of data and three columns of "ID", "TEXT", and "LABEL". Each datapoint ID came pre-labeled with the "0, 1, 2" categories. The 0 category, "Not a Movie/TV Show Review", was the largest and contained 32,071 of IDs. The actual reviews, 1 (Positive Review) and 2 (Negative Review), contained 19,276 and 18,970 IDs, respectively. The "test.csv" contained 17,580 rows of data split into two columns, "ID" and "TEXT".

A quick review of the training data yielded the following observations. One, there was a significantly larger amount of "0" labels than there were "1" or "2" labels. Additionally, the "train.csv" dataset contained text that should not have been used to train the model as it would have impacted the success of predictions for "test.csv". For example, there was data containing a language different to that of the target language (English), such as ID 11352924827579021872 which contained "491. Из 9 фунтов муки испечено 16 белых хлебов. Сколько муки пошло на каждый?". For testing data, I also had a similar issue with data that would have impacted the model's success. The "test.csv" contained texts in languages other than English, such as ID 6577057911690817390, which contained "對看講出無量法，老幼人人放喜懷". While reading in the .csvs using pandas, I also noticed a review that contained emojis, another type of special character that could impact model success. I opened and formatted the .csv files for analysis using the below code:

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

Once I had finished analyzing and formatting both sets of data, I was able to move into the steps for training the model. 

### 2. Training the Model

To train the model, I first needed to create vectors for the labels (0,1,2) and text from the training data. I made use of both *CountVectorizer* and *LabelEncoder* in order to convert the "TEXT" columns into feature vectors that could be recognized by the fit/predict aspects of the LogisticRegression classifier as well as to convert the "LABEL" columns into a label vector that would assign new values to our "test.csv" IDs in the future. Please see below for how I created the vectors:

```python
#creating features from the training data text
cv = CountVectorizer()
x = cv.fit_transform(data['TEXT'].apply(lambda x: np.str_(x)))

#creating labels from the training data text
le = LabelEncoder()
y = le.fit_transform(data['LABEL'])
```

Once the vectors had been created I could train my model with them, adjusting the LogisticRegression parameters as necessary. I bumped up the default value for max_iter, which is 100, to 10,000 and added random_state set to 42. Depending on your solver of choice and data, you need to adjust the max_iter to get your solver to converge and reach the optimal solution. Setting random_state to 42 allows the dataset to be shuffled and is a standard value to use (along with 0). Training the model and updating parameters is a pretty quick process thanks to scikit-learn as demonstrated here:

```python
#train the model
LR = LogisticRegression(max_iter=10000, random_state=42)
model = LR.fit(x,y)
```

### 3. Analyzing Test Data

Once the model had been fit successfully, I could create a new feature vector for a 'TEXT' column that represents the test data. This process mirrors the one for the training data above:

```python
#creating features from the test data text
cv2 = CountVectorizer()
x2 = cv.transform(data2['TEXT'].apply(lambda x: np.str_(x)))
```

### 4. Predictions & Outputs 

Now that I had a trained model and final feature vector, it was time to check how well the model performed. To create predictions for the new feature vector, I ran the following code: 

```python
#get predictions
predictions = LR.predict(x2)

#to check the shape and length of the returned vector, uncomment the following:
#print(predictions)
#print(len(predictions))
```

To analyze the predictions, I output them into a .csv file for easier viewing. Pandas DataFrames simplified this process of outputting and saving my prediction results, as I could simply create a new .csv with headers and columns of my choice:

```python
#creating .csv
csv = pd.DataFrame({"ID":list(data2["ID"]), "LABEL":list(predictions)})

#outputting it to my desktop for saving and sharing outside of Jupyter
csv.to_csv(r"C:\Users\Alex\Downloads\final3.csv",index=False)
```

### 5. Results & Future Improvements 

The final iteration of my model achieved an F1 score just above 90%. I was pleased to see that all non-English reviews were correctly categorized as "0", indicating that the model successfully rejected any off-topic content. 

To further improve my model's performance, I have identified three key areas for improvement. One, as briefly touched on a few sections above, is to refine the training dataset by removing data points that differ from the target dataset. This would include non-English text, gibberish, or other irrelevant content. This would be possible to implement through tokenization and normalization techniques--such as regular expressions or tools from NLTK, a natural language processing library. Removing and normalizing the data would ensure that the feature vectors created are more representative of the target data, resulting in more meaningful patterns identified. Similarly, it would also be a great idea to confirm that the labels of the training data are correct before using them to train the model. A second change would be to adjust my LogisticRegression parameters to check if other multinomial solvers, such as newton-cg or sag, would be better choices than the default one, lbfgs. Alternatively, an ideal approach could be to utilize *TFIDFVectorizer*, which would assist in weighting the frequency of the words, as opposed to *CountVectorizer*. Additionally, I could build a confusion matrix to conduct more thorough error analysis. This matrix would allow me to break down the model’s predictions into true positives, false positives, false negatives, and true negatives—offering insight into where misclassifications occur. By reviewing these in conjunction with the actual text inputs, I could better understand error patterns and make targeted adjustments to improve the model's classification of edge cases.    





