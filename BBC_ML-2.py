
# coding: utf-8

# In[19]:


import pandas as pd
import numpy as np

data = pd.read_csv("/Users/vineevineela/Desktop/bbc/dataset.csv")


# In[20]:


data.shape


# In[21]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data['news'],data['type'],test_size=0.25)


# In[22]:


X_train.shape


# In[23]:


X_test.shape


# In[24]:


train_data = pd.concat([X_train,y_train],axis=1,ignore_index=True)
train_data.rename(columns={0:'news', 1:'type'}, inplace=True)
train_data.head(5)


# In[25]:


test_data = pd.concat([X_test,y_test],axis=1,ignore_index=True)
test_data.rename(columns={0:'news', 1:'type'}, inplace=True)
test_data.head(5)
test_data.to_csv("/Users/vineevineela/Desktop/bbc/test_data.csv")


# In[26]:


#preprocessing of train data
train_data['news'] = map(lambda x: x.lower(), train_data['news'])#lowercase
train_data['news'] = train_data['news'].str.replace('\d+', '') #all numbers are replaced
train_data['news']= train_data['news'].str.replace(r"\(.*\)","") # remove paranthesis data
train_data['news'] = train_data['news'].str.replace('[^\w\s]','') # remove punctuations
train_data['news'] = train_data['news'].str.findall('\w{3,}').str.join(' ') #remove small words

train_data['news'] = train_data['news'].astype("string")


# In[27]:


#to tokenize into words
from nltk.tokenize import word_tokenize
train_data['words'] = train_data['news'].apply(word_tokenize)

#to remove stopwords
from nltk.corpus import stopwords
stop = stopwords.words('english')
train_data['Swords'] = train_data['words'].apply(lambda x: [item for item in x if item not in stop])

#stemming
#from nltk.stem import PorterStemmer
#ps = PorterStemmer()
#train_data['Stabstract'] = train_data['Sabstract'].apply(lambda x: [ps.stem(y) for y in x])

#lemmetizing
from nltk.stem.wordnet import WordNetLemmatizer
lmtzr = WordNetLemmatizer()
train_data['Lwords'] = train_data['Swords'].apply(lambda x: [lmtzr.lemmatize(y) for y in x])

#joining
train_data['Jwords'] = train_data['Lwords'].apply(' '.join)


# In[28]:


train_data.head(5)


# In[41]:



train_data['Jwords'].shape


# In[42]:


import cPickle as cp
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
cv = cv.fit(train_data['Jwords'])
cv_traindata = cv.transform(train_data['Jwords'])
#cv_testdata = cv.transform(test_data['Jwords'])
p = cp.dump(cv,open("/Users/vineevineela/Desktop/bbc/count.p","wb"))
cv_traindata.shape
print p


# In[37]:


from sklearn.feature_extraction.text import TfidfTransformer
tf = TfidfTransformer()
tf = tf.fit(cv_traindata)
tf_traindata = tf.transform(cv_traindata)
q = cp.dump(tf,open("/Users/vineevineela/Desktop/bbc/tf.p","wb"))
#tf_testdata = tf.transform(cv_testdata)


# In[43]:


import time
from sklearn.neural_network import MLPClassifier

start_time = time.clock()
mlp = MLPClassifier(solver='sgd', hidden_layer_sizes=(15,), random_state=1)
mlp = mlp.fit(tf_traindata,train_data['type'])
pred_mlp = mlp.predict(tf_testdata)

from sklearn.metrics import accuracy_score
MLP_Accuracy = accuracy_score(test_data['type'], pred_mlp)
print("MLP Accuracy = ",MLP_Accuracy)

from sklearn.metrics import f1_score
f1_score = f1_score(test_data['type'], pred_mlp, average= 'weighted') 
print ("MLP F1Score = ",f1_score)

print("Time Taken in seconds:", time.clock() - start_time) 


# In[18]:


#SGD Classifier

from sklearn.linear_model import SGDClassifier

start_time = time.clock()
SGD_model = SGDClassifier().fit(tf_traindata,train_data['type'])
pred_sgd = SGD_model.predict(tf_testdata)

from sklearn.metrics import accuracy_score
SGD_Accuracy = accuracy_score(test_data['type'], pred_sgd)
print("SGD Accuracy =",SGD_Accuracy)

from sklearn.metrics import f1_score
SGD_F1 = f1_score(test_data['type'], pred_sgd, average= 'weighted') 
print("SGD F1Score =",SGD_F1)
 
print("Time Taken in seconds:", time.clock() - start_time) 


# In[20]:


#decision tree

from sklearn import tree
from sklearn.metrics import accuracy_score

start_time = time.clock()
c1 = tree.DecisionTreeClassifier()
c1.fit(tf_traindata,train_data['type'])
Dectree_Y =c1.predict(tf_testdata)

from sklearn.metrics import accuracy_score
Dec_Accuracy = accuracy_score(test_data['type'], Dectree_Y)
print("Decision Tree Accuracy = ",Dec_Accuracy)

from sklearn.metrics import f1_score
Dec_F1 = f1_score(test_data['type'],Dectree_Y, average= 'weighted')
print("Decision Tree F1Score = ",Dec_F1)

print("Time Taken in seconds:", time.clock() - start_time) 


# In[21]:


#GradientBoost

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score

start_time = time.clock()
c2 = GradientBoostingClassifier(n_estimators=500)
c2.fit(tf_traindata,train_data['type'])
GradB_Y = c2.predict(tf_testdata)

from sklearn.metrics import accuracy_score
Grad_Accuracy = accuracy_score(test_data['type'], GradB_Y)
print("Gradient Boost accuracy = ",Grad_Accuracy)

from sklearn.metrics import f1_score
Grad_F1 = f1_score(test_data['type'],GradB_Y, average= 'weighted')
print("Gradient Boost F1Score = ",Grad_F1)

print("Time Taken in seconds:", time.clock() - start_time) 


# In[24]:


#SVM Classifier

from sklearn.svm import SVC

start_time = time.clock()

SVC_model = SVC().fit(tf_traindata,train_data['type'])
pred_svc = SVC_model.predict(tf_testdata)

from sklearn.metrics import accuracy_score
SVC_Accuracy = accuracy_score(test_data['type'], pred_sgd)
print("SVM Accuracy=",SVC_Accuracy)

from sklearn.metrics import f1_score
SVC_F1 = f1_score(test_data['type'], pred_sgd, average= 'weighted') 
print("SVM F1Score=",SVC_F1)
 
print("Time Taken in seconds:", time.clock() - start_time) 

