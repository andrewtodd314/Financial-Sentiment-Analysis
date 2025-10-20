#!/usr/bin/env python
# coding: utf-8

import torch
import numpy as np
import gradio as gr
from transformers import AutoTokenizer, AutoModel
from tensorflow.keras.models import load_model

# Load FinBERT model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
finbert_model = AutoModel.from_pretrained("yiyanghkust/finbert-tone")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
finbert_model.to(device)

# Function to get CLS embeddings
def get_cls_embeddings(texts, batch_size=1):
    finbert_model.eval()
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
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
            outputs = finbert_model(input_ids=input_ids, attention_mask=attention_mask)
            cls_embeddings = outputs.last_hidden_state[:, 0, :]  # [CLS] token
            all_embeddings.append(cls_embeddings.cpu().numpy())

    return np.vstack(all_embeddings)

# Load your Keras model
model_text = load_model("model_text.h5")

# Prediction function
def predict_sentiment(user_input):
    if not user_input.strip():
        return "Please enter financial text to analyze."

    X_embeddings = get_cls_embeddings([user_input])
    probs = model_text.predict(X_embeddings)

    class_labels = ["neutral", "positive", "negative"]
    predicted_index = np.argmax(probs)
    predicted_label = class_labels[predicted_index]
    predicted_confidence = probs[0][predicted_index]

    return f"**Predicted Sentiment:** {predicted_label.capitalize()} ({predicted_confidence:.2f})"

# Gradio Interface
title = "💹 Financial Sentiment Classifier"
description = """
This app uses **FinBERT** embeddings and a trained Keras model to classify financial text into **positive**, **neutral**, or **negative** sentiment.
"""

iface = gr.Interface(
    fn=predict_sentiment,
    inputs=gr.Textbox(lines=6, placeholder="Enter financial text here...", label="Financial Text"),
    outputs=gr.Markdown(label="Prediction"),
    title=title,
    description=description,
    examples=[
        ["The company's earnings exceeded analyst expectations, leading to a strong market reaction."],
        ["Revenue declined this quarter due to weaker demand."],
        ["The performance remained stable with no major surprises."]
    ],
)

iface.launch(share=True)

