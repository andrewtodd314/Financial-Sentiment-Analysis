#!/usr/bin/env python
# coding: utf-8

# # Creating the model that will be used for classifying financial sentences on Streamlit API

# In[8]:


import requests
import pandas as pd
import re
import numpy as np
import torch
import transformers as ppb
from numpy import linspace
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
#from wordcloud import WordCloud
import seaborn as sns
import pysentiment2 as ps
from keras.models import model_from_json
from keras.layers import Dense
from keras.models import Sequential
#from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping
#from keras.optimizers import SGD doesnt work
import scikeras
from scikeras.wrappers import KerasClassifier
import keras.optimizers
import os
import tensorflow as tf
#from keras.optimizers import RMSprop doesnt work
#from prettytable import PrettyTable
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import preprocessing
from tensorflow.keras.optimizers import RMSprop
import pickle
import torch
import numpy as np
import pandas as pd
from transformers import BertTokenizer, BertModel
from tqdm.notebook import tqdm  # better for Jupyter
import streamlit as st


# In[9]:


#Setting up the apps title
st.title("💹 Financial Sentiment Classifier")


# In[ ]:


#take in the apps input
user_input = st.text_area("Enter financial text:", "")


# In[ ]:


# Load FinBERT model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
model = AutoModel.from_pretrained("yiyanghkust/finbert-tone")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def get_cls_embeddings(texts, batch_size=4):
    model.eval()
    all_embeddings = []

    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i:i + batch_size]

        encoded = tokenizer.batch_encode_plus(
            batch,
            add_special_tokens=True,
            padding="longest",
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )

        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            cls_embeddings = outputs.last_hidden_state[:, 0, :]  # [CLS] token
            all_embeddings.append(cls_embeddings.cpu().numpy())

    return np.vstack(all_embeddings)


# In[ ]:


#take in the apps input
user_input = st.text_area("Enter financial text:", "")


# In[ ]:


#generate embeddings for Xtrain and Xtest
X_embeddings = get_cls_embeddings([User_input], batch_size=4)


# In[ ]:


# Loading model
os.chdir('C:/Users/xeb15154/OneDrive - University of Strathclyde/Financial Sentiment Analysis Project/')
with open("model_text.pkl", "rb") as f:
    model_text = pickle.load(f)


# In[ ]:


probs = model_text.predict(X_embeddings)

class_labels = ["neutral","positive","negative"]

predicted_index = np.argmax(probs)
predicted_label = class_labels[predicted_index]
predicted_confidence = probs[0][predicted_index]
print(f"Predicted Sentiment: {predicted_label} ({predicted_confidence:.2f})")

