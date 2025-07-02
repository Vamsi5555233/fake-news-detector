import streamlit as st
import pickle

# Load the model and vectorizer
with open("final_model.sav", "rb") as model_file:
    model = pickle.load(model_file)

with open("vectorizer.pkl", "rb") as vec_file:
    vectorizer = pickle.load(vec_file)

# Set custom page config
st.set_page_config(
    page_title="Fake News Detector by Vamsi Krishna",
    page_icon="📰",
    layout="centered"
)

# --- Custom CSS Styling ---
st.markdown("""
    <style>
        body {
            background-color: #f9f9f9;
            font-family: 'Segoe UI', sans-serif;
        }
        .title {
            font-size: 3em;
            font-weight: bold;
            color: #333333;
        }
        .footer {
            text-align: center;
            margin-top: 2rem;
            color: gray;
            font-size: 0.9em;
        }
        .stTextArea > label {
            font-weight: 600;
        }
        .stButton>button {
            background-color: #ff4b4b;
            color: white;
            font-weight: bold;
            border-radius: 5px;
            padding: 0.5em 1.5em;
        }
        .stButton>button:hover {
            background-color: #ff3333;
        }
    </style>
""", unsafe_allow_html=True)

# --- App UI ---
st.markdown("<div class='title'>📰 Fake News Detector</div>", unsafe_allow_html=True)
st.write("Enter the news text below to check if it's **real or fake**:")

# --- User Input ---
user_input = st.text_area("News Text", height=150)

# --- Prediction Button ---
if st.button("Predict"):
    if not user_input.strip():
        st.warning("⚠️ Please enter some text.")
    else:
        # Transform and predict
        vectorized_input = vectorizer.transform([user_input])
        prediction = model.predict(vectorized_input)[0]

        # Labels and Emojis
        label_colors = {
            "true": "🟢 Real News",
            "false": "🔴 Fake News",
            "pants-fire": "🔥 Pants-Fire",
            "half-true": "🟡 Half-True",
            "barely-true": "🟠 Barely-True",
            "mostly-true": "🔵 Mostly-True",
            "fake": "❌ Totally Fake"
        }

        result = label_colors.get(prediction.lower(), prediction)

        st.markdown(f"""
            <h3>Prediction:</h3>
            <div style='background-color:#e6ffee;padding:10px;border-radius:10px;margin-top:10px;font-size:1.3em'>
                {result}
            </div>
        """, unsafe_allow_html=True)

# --- Footer ---
st.markdown("<div class='footer'>👨‍💻 Built with ❤️ by <b>Vamsi Krishna</b></div>", unsafe_allow_html=True)
