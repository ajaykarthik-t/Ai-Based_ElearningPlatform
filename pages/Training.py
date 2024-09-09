import streamlit as st
import pandas as pd
import numpy as np
import re
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

st.title('Training Page')

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text

def train_model():
    data = pd.read_csv(st.file_uploader("Upload your CSV file", type="csv"))
    if data is not None:
        st.write("Data loaded successfully!")
        data = data.drop(columns=['image'])
        data.dropna(subset=['question', 'answer'], inplace=True)
        data.reset_index(drop=True, inplace=True)

        data['processed_question'] = data['question'].apply(preprocess_text)
        data['processed_answer'] = data['answer'].apply(preprocess_text)

        category_encoder = LabelEncoder()
        data['category_encoded'] = category_encoder.fit_transform(data['question_type'])

        tokenizer = Tokenizer(num_words=5000)
        tokenizer.fit_on_texts(data['processed_question'])
        X = tokenizer.texts_to_sequences(data['processed_question'])
        X = pad_sequences(X, maxlen=50)

        y = data['category_encoded']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = Sequential()
        model.add(Embedding(input_dim=5000, output_dim=128, input_length=50))
        model.add(LSTM(128, return_sequences=True))
        model.add(LSTM(64))
        model.add(Dense(len(data['question_type'].unique()), activation='softmax'))
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        history = model.fit(X_train, np.array(y_train), epochs=20, validation_data=(X_test, np.array(y_test)))

        # Save model and tokenizer
        model.save('model.h5')
        with open('tokenizer.pickle', 'wb') as handle:
            pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open('category_encoder.pickle', 'wb') as handle:
            pickle.dump(category_encoder, handle, protocol=pickle.HIGHEST_PROTOCOL)

        st.success("Model trained and saved successfully!")

train_model()
