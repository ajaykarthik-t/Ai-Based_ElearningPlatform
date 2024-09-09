import streamlit as st
import pandas as pd
import re
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from fuzzywuzzy import fuzz

st.title('Inference Page')

# Function Definitions
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)  # Remove digits
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return text

def predict_category(question):
    processed_question = preprocess_text(question)
    sequence = tokenizer.texts_to_sequences([processed_question])
    padded_sequence = pad_sequences(sequence, maxlen=50)
    prediction = model.predict(padded_sequence)
    predicted_category = np.argmax(prediction, axis=1)[0]
    return category_encoder.inverse_transform([predicted_category])[0]

def collect_responses(questions_df):
    responses = []
    for i, row in questions_df.iterrows():
        user_answer = st.text_input(f"Question {i+1}: {row['question']}", key=i)
        if user_answer:
            responses.append({
                'question': row['question'],
                'user_answer': user_answer,
                'correct_answer': row['answer']
            })
    return pd.DataFrame(responses)

def display_summary(user_responses):
    user_responses['score'] = user_responses.apply(lambda row: 1 if row['user_answer'].strip().lower() == row['correct_answer'].strip().lower() else 0, axis=1)
    correct_count = user_responses['score'].sum()
    total_count = len(user_responses)
    
    st.write(f"**Summary:**")
    st.write(f"Total Questions: {total_count}")
    st.write(f"Correct Answers: {correct_count}")
    st.write(f"Incorrect Answers: {total_count - correct_count}")
    st.write(f"Score: {correct_count/total_count * 100:.2f}%")
    
    st.bar_chart([correct_count, total_count - correct_count], labels=['Correct', 'Incorrect'])

def save_responses(user_responses, file_name='user_responses.csv'):
    user_responses = user_responses[['question', 'user_answer', 'correct_answer']]
    user_responses.to_csv(file_name, index=False)
    st.success(f"User responses saved to {file_name}")

def provide_suggestions(user_responses):
    incorrect_answers = user_responses[user_responses['score'] == 0]
    if not incorrect_answers.empty:
        st.write("**Suggestions for Improvement:**")
        for index, row in incorrect_answers.iterrows():
            st.write(f"- Review the question: '{row['question']}'")
    else:
        st.write("Congratulations! You answered all questions correctly. Keep up the good work!")

def calculate_similarity(user_answer, correct_answer):
    return fuzz.ratio(user_answer.lower(), correct_answer.lower())

def recommend_focus(user_responses):
    recommendations = []
    for index, row in user_responses.iterrows():
        similarity = calculate_similarity(row['user_answer'], row['correct_answer'])
        if similarity < 80:  # Consider answers with less than 80% similarity as incorrect
            recommendations.append({
                'question': row['question'],
                'user_answer': row['user_answer'],
                'correct_answer': row['correct_answer'],
                'similarity': similarity
            })
    
    recommendations_df = pd.DataFrame(recommendations)
    if not recommendations_df.empty:
        st.write("**Focus Areas for Spelling Improvement:**")
        for index, row in recommendations_df.iterrows():
            st.write(f"- For question '{row['question']}', your answer was '{row['user_answer']}'.")
            st.write(f"  Correct answer is '{row['correct_answer']}'. Similarity score: {row['similarity']}")
    else:
        st.write("No significant spelling errors detected. Good job!")

# Load model, tokenizer, and category encoder
model = load_model('model.h5')
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
with open('category_encoder.pickle', 'rb') as handle:
    category_encoder = pickle.load(handle)

# User input for categories
st.write("**Select Categories:**")
all_categories = [
    "General Knowledge For Kids",
    "GK Questions For Class 1",
    "GK Questions For Class 2",
    "GK Questions For Class 3",
    "GK Questions For Class 4",
    "GK Questions For Class 5",
    "GK Questions For Class 6",
    "GK Questions For Class 7"
]
selected_categories = st.multiselect("Choose categories", all_categories)

# Upload a CSV file
uploaded_file = st.file_uploader("Upload a CSV file with questions", type="csv")

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    data = data.drop(columns=['image'])
    data.dropna(subset=['question', 'answer'], inplace=True)
    data.reset_index(drop=True, inplace=True)
    
    # Button to proceed to the next step
    if st.button("Submit Categories"):
        if selected_categories:
            filtered_questions = data[data['question_type'].isin(selected_categories)]
            num_questions = min(10, len(filtered_questions))
            selected_questions = filtered_questions.sample(n=num_questions).reset_index(drop=True)
            
            st.write("**Selected Questions:**")
            st.write(selected_questions[['question']])
            
            # Collect responses
            responses_df = collect_responses(selected_questions)
            
            if st.button("Submit Answers"):
                if responses_df.shape[0] == 10:  # Ensure 10 answers are provided
                    display_summary(responses_df)
                    save_responses(responses_df)
                    provide_suggestions(responses_df)
                    recommend_focus(responses_df)
                else:
                    st.warning("Please answer all 10 questions before submitting.")
        else:
            st.warning("Please select at least one category before submitting.")
