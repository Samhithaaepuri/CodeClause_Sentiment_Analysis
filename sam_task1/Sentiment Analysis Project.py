#!/usr/bin/env python
# coding: utf-8

# # Sentiment Analysis

# In[1]:


#importing pandas and numpy
import numpy as np
import pandas as pd


# In[2]:


#importing training dataset 
df= pd.read_csv('sentiment_train.csv')


# In[3]:


df.sample(4)


# In[4]:


df.shape


# In[5]:


#summary 
df.info()


# In[6]:


#check for missing values
df.isnull().sum()


# In[7]:


#renaming the columns
df.rename(columns={'Sentence':'text','Polarity':'target'},inplace=True)


# In[8]:


df.head()


# In[9]:


#check for duplicate values
df.duplicated().sum()


# In[10]:


#remove duplicates
df = df.drop_duplicates(keep='first')


# In[11]:


df.duplicated().sum()


# In[12]:


df.shape


# In[13]:


df.head()


# In[14]:


df['target'].value_counts()


# In[15]:


#importing matplotlib for visualization
import matplotlib.pyplot as plt
plt.pie(df['target'].value_counts(), labels=['positive','negative'], autopct="%0.2f")
plt.show()


# In[16]:


# Data is balanced


# In[17]:


get_ipython().system('pip install nltk')


# In[18]:


import nltk


# In[19]:


nltk.download('punkt')


# In[20]:


# number of characters in text
df['num_characters'] = df['text'].apply(len)


# In[21]:


df.head()


# In[22]:


# number of words
df['num_words'] = df['text'].apply(lambda x:len(nltk.word_tokenize(x)))


# In[23]:


df.head()


# In[24]:


df['num_sentences'] = df['text'].apply(lambda x:len(nltk.sent_tokenize(x)))


# In[25]:


df.head()


# In[26]:


df[['num_characters','num_words','num_sentences']].describe()


# In[27]:


# positive reviews
df[df['target'] == 1][['num_characters','num_words','num_sentences']].describe()


# In[28]:


# negative reviews
df[df['target'] == 0][['num_characters','num_words','num_sentences']].describe()


# In[29]:


#importing seaborn for visualization
import seaborn as sns


# In[30]:


plt.figure(figsize=(9,6))
sns.histplot(df[df['target'] == 1]['num_characters'])
sns.histplot(df[df['target'] == 0]['num_characters'],color='red')


# In[31]:


plt.figure(figsize=(9,6))
sns.histplot(df[df['target'] == 1]['num_words'])
sns.histplot(df[df['target'] == 0]['num_words'],color='red')


# In[32]:


sns.pairplot(df,hue='target')


# In[33]:


#feature correlation
sns.heatmap(df.corr(), annot=True)


# In[34]:


# importing testing dataset
test= pd.read_csv('sentiment_test.csv')


# In[35]:


test.sample(4)


# In[36]:


test.shape


# In[37]:


#summary test data
test.info()


# In[38]:


#check for missing values
test.isnull().sum()


# In[39]:


#renaming the columns
test.rename(columns={'Sentence':'text','Polarity':'target'},inplace=True)


# In[40]:


test.head()


# In[41]:


#check for duplicate values
test.duplicated().sum()


# In[42]:


#remove duplicates
test = test.drop_duplicates(keep='first')


# In[43]:


test.duplicated().sum()


# In[44]:


test.shape


# In[45]:


test.head()


# In[46]:


test['target'].value_counts()


# In[47]:


#number of characters in text
test['num_characters'] = test['text'].apply(len)


# In[48]:


test.head()


# In[49]:


# number of words
test['num_words'] = test['text'].apply(lambda x:len(nltk.word_tokenize(x)))


# In[50]:


test.head()


# In[51]:


# num of sentence
test['num_sentences'] = test['text'].apply(lambda x:len(nltk.sent_tokenize(x)))


# In[52]:


test.head()


# In[54]:


#Data Preprocessing


# In[56]:


import nltk


# In[57]:


nltk.download('stopwords')


# In[58]:


#import nltk for text processing
from nltk.corpus import stopwords
stopwords.words('english')


# In[59]:


#import string
import string
string.punctuation


# In[60]:


#import porterstemmer
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
ps.stem('roaming')


# In[61]:


# function for text preprocessing 
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    
    text = y[:]
    y.clear()
    
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
            
    text = y[:]
    y.clear()
    
    for i in text:
        y.append(ps.stem(i))
    
            
    return " ".join(y)


# In[62]:


#testing the function
transform_text('Hi how are you Pranav?')


# In[63]:


df.head()


# In[64]:


# applying the text transformation to our training data
df['text'].apply(transform_text)


# In[65]:


df['transform_text'] = df['text'].apply(transform_text)


# In[66]:


df.head()


# In[67]:


# applying the text transformation to our testing data
test['transform_text'] = test['text'].apply(transform_text)


# In[68]:


test.head()


# In[69]:


#Model Building


# In[70]:


#text Vectorization
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()


# In[71]:


X_train = cv.fit_transform(df['transform_text']).toarray()
X_test = cv.transform(test['transform_text']).toarray()


# In[72]:


X_train.shape


# In[73]:


y_train=df['target'].values


# In[74]:


y_train.shape


# In[75]:


X_test.shape


# In[76]:


y_test=test['target'].values


# In[77]:


y_test.shape


# In[78]:


#Trying different Naive bayes nodels


# In[79]:


#importing NB models
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score


# In[80]:


gnb = GaussianNB()
mnb = MultinomialNB()
bnb = BernoulliNB()


# In[81]:


# Gaussian NB model accuracy, precision, f1, roc_auc_score

gnb.fit(X_train,y_train)
y_pred1 = gnb.predict(X_test)
print('Accuracy ',accuracy_score(y_test,y_pred1))
print(confusion_matrix(y_test,y_pred1))
print(precision_score(y_test,y_pred1))
print(recall_score(y_test,y_pred1))
print(f1_score(y_test,y_pred1))
print(roc_auc_score(y_test,y_pred1))


# In[82]:


# Multinomial NB model accuracy, precision, f1, roc_auc_score

mnb.fit(X_train,y_train)
y_pred2 = mnb.predict(X_test)
print(accuracy_score(y_test,y_pred2))
print(confusion_matrix(y_test,y_pred2))
print(precision_score(y_test,y_pred2))
print(recall_score(y_test,y_pred2))
print(f1_score(y_test,y_pred2))
print(roc_auc_score(y_test,y_pred2))


# In[83]:


# Bernoulli NB model accuracy, precision, f1, roc_auc_score

bnb.fit(X_train,y_train)
y_pred3 = bnb.predict(X_test)
print(accuracy_score(y_test,y_pred3))
print(confusion_matrix(y_test,y_pred3))
print(precision_score(y_test,y_pred3))
print(recall_score(y_test,y_pred3))
print(f1_score(y_test,y_pred3))
print(roc_auc_score(y_test,y_pred3))


# In[84]:


#Using Random Forest Classifier


# In[85]:


#importing randomforest classifier
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=200,random_state=0)


# In[86]:


rfc.fit(X_train, y_train)
y_pred4 = rfc.predict(X_test)
print(accuracy_score(y_test,y_pred4))
print(confusion_matrix(y_test,y_pred4))
print(precision_score(y_test,y_pred4))
print(recall_score(y_test,y_pred4))
print(f1_score(y_test,y_pred4))
print(roc_auc_score(y_test,y_pred4))


# In[87]:


#importing GridsearchCV
from sklearn.model_selection import GridSearchCV


# In[88]:


param_grid = { 
    'n_estimators': [200, 500],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth' : [4,5,6,7,8],
    'criterion' :['gini', 'entropy']
}


# In[ ]:





# In[90]:


CV_rfc.best_params_


# In[93]:


rfc1=RandomForestClassifier(random_state=0, max_features='auto', n_estimators= 500, max_depth=8, criterion='entropy')


# In[94]:


rfc1.fit(X_train, y_train)
y_pred5 = rfc1.predict(X_test)
print(accuracy_score(y_test,y_pred5))
print(confusion_matrix(y_test,y_pred5))
print(precision_score(y_test,y_pred5))
print(recall_score(y_test,y_pred5))
print(f1_score(y_test,y_pred5))
print(roc_auc_score(y_test,y_pred5))


# In[95]:


from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix


# In[96]:


#generating report for random forest
rfc_cv_score = cross_val_score(rfc, X_train, y_train, cv=10, scoring='roc_auc')


# In[97]:


print("=== Confusion Matrix ===")
print(confusion_matrix(y_test, y_pred4))
print('\n')
print("=== Classification Report ===")
print(classification_report(y_test, y_pred4))
print('\n')
print("=== All AUC Scores ===")
print(rfc_cv_score)
print('\n')
print("=== Mean AUC Score ===")
print("Mean AUC Score - Random Forest: ", rfc_cv_score.mean())


# In[2]:


# generating report after cross validation
#rfc1_cv_score = cross_val_score(rfc1, X_train, y_train, cv=10, scoring='roc_auc')


# In[99]:


print("=== Confusion Matrix ===")
print(confusion_matrix(y_test, y_pred5))
print('\n')
print("=== Classification Report ===")
print(classification_report(y_test, y_pred5))
print('\n')
print("=== All AUC Scores ===")
print(rfc1_cv_score)
print('\n')
print("=== Mean AUC Score ===")
print("Mean AUC Score - Random Forest: ", rfc1_cv_score.mean())

