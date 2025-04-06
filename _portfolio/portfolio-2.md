---
title: "Multi-Class Classifier"
excerpt: "A classifier that utilizes logistic regression to categorize TV/movie reviews into 3 categories. ![Internship Post Image](/images/classifier_image.png)"
collection: portfolio
---
_Tools & Technologies: Pandas, NumPy, scikit-learn_

### Summary

The goal of my project was to
I decided to use logistic regression because it was the alogrithm that I had the most experience with, and 

In order to get predictions, I had to analyze the data, fit the model to training data, and analyze the test data using the fitted model.  

### 1. Analyzing Test Data



### 2. Training the Model


Please see below for how I created the vectors:
```python
#creating features from the text
cv = CountVectorizer()
x = cv.fit_transform(data['TEXT'].apply(lambda x: np.str_(x)))

#creating labels from the text
le = LabelEncoder()
y = le.fit_transform(data['LABEL'])
```

Once the vectors have been created, we can train our model using those vectors and adjust the LogisticRegression parameters as necessary. 


```python
#train the model
LR = LogisticRegression(max_iter=10000, random_state=42)
model = LR.fit(x,y)
```

### 3. Analyzing Test Data

### 4. Predictions & Outputs 

```python
#get predictions
predictions = LR.predict(x2)

#to check the shape and length of the returned vector, uncomment the following:
#print(predictions)
#print(len(predictions))
```

```python
#creating .csv
csv = pd.DataFrame({"ID":list(data2["ID"]), "LABEL":list(predictions)})

#output .csv

```



