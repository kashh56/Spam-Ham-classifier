#!/usr/bin/env python
# coding: utf-8

# ## Spam Mail Classifier 

# #

# overview , objective , methodology

# #

# #### Binary classification using logisitc regression

# In[75]:


#Importing the required libraries

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score 
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer



# In[76]:


#Reading the csv and seeing top 5 rows

data = pd.read_csv('mail_data.csv')
data.head()


# In[77]:


#lets convert Category columns value to numerical (ham=1 , spam=0)
data['Category'] = data['Category'].map({'ham':1 , 'spam':0})


# In[78]:


#lets split the data into X and y

X = data['Message']
y = data['Category']


# In[79]:


# making our train and test split
X_train , X_test , y_train , y_test = train_test_split(X,y,random_state=100 , train_size=0.80)


# In[80]:


# lets also convert the Messages column to numeric values using TFid vectorizer , 
 
# TF-IDF Vectorizer (Term Frequency-Inverse Document Frequency) 
# is a popular technique in text mining and natural language processing (NLP) to convert textual data into numerical features. It reflects how 
# important a word is to a document in a collection (or corpus). This importance increases with the number of times the 
# word appears in the document but is offset by how frequently the word occurs in the entire corpus. 


# In[81]:


# Step 1: Initialize the TfidfVectorizer
vectorizer = TfidfVectorizer(stop_words='english')


# In[82]:


# Step 2: Fit the vectorizer and transform the training data
X_train_features = vectorizer.fit_transform(X_train)


# In[83]:


# Step 3: Transform the test data
X_test_features = vectorizer.transform(X_test)


# In[84]:


# lets check the percentage of spam and ham email     
data.Category.value_counts(normalize=True) * 100


# In[85]:


# we notice that the data is highly imbalanced to fix this we will need to adjust the hyper-parameter class_weight inside LogisticRegression
# function


# In[86]:


# Train a Logistic Regression model

model = LogisticRegression(class_weight='balanced')
model.fit(X_train_features, y_train)


# In[87]:


# Make predictions on the test data
y_test_pred = model.predict(X_test_features)


# In[88]:


# accuracy score of test set
accuracy_score(y_test_pred,y_test)


# In[89]:


# 1 is ham , 0 is spam 

def spam_ham_checker(abc):
    try:
        # Transform input using the vectorizer
        X_feature = vectorizer.transform([abc])
        prediction = model.predict(X_feature)

        # Log prediction for debugging
        print(f"DEBUG: Model prediction: {prediction}")

        # Return result based on prediction
        return 'ham' if prediction[0] == 1 else 'spam mail'
    except Exception as e:
        print(f"Error during prediction: {e}")
        return None  # Explicitly return None on error

        


# In[90]:


spam_ham_checker("Tell where you reached")


# In[91]:


data[data.Category == 1]['Message'].head(50)


# In[92]:


# spam 
#  URGENT! Your Mobile No. was awarded £2000

