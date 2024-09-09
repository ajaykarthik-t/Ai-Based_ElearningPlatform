import streamlit as st

st.set_page_config(page_title="General Knowledge Quiz App", layout="wide")

st.title("General Knowledge Quiz App")

# Add navigation sidebar
st.sidebar.title("Navigation")
selected_page = st.sidebar.radio("Go to", ["Training", "Inference"])

if selected_page == "Training":
    st.write("## Training Page")
    st.markdown("Navigate to the **Training** page to train the model.")
elif selected_page == "Inference":
    st.write("## Inference Page")
    st.markdown("Navigate to the **Inference** page to use the trained model for predictions.")
