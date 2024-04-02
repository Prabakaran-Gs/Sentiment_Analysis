import streamlit as st 
import joblib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def predict(raw_test: str) -> tuple[str, list[float]]:
    """
    Predict the emotion of a given sentence.

    Args:
        raw_test (str): The sentence to be classified.

    Returns:
        tuple[str, list[float]]: The predicted emotion and its probability.
    """
    classes = ["Sad", "Joy", "Love", "Anger", "Fear", "Surprise"]

    # load model
    model = joblib.load("model.pkl")

    # Predict the ouput class
    output_id = model.predict([raw_test])[0]

    # Predict the probs
    probs = model.predict_proba([raw_test])
    ouput = classes[output_id]

    return ouput, probs

def main():
    st.title("Emotion Classifier Using Text")

    col1, col2 = st.columns(2)
    with col1:
        st.write("Prabakaran GS")
    with col2:
        st.page_link("https://github.com/prabakaran-Gs",label='Github')

    st.subheader("Enter a sentence:")
    raw_test = st.text_input("Text :")

    if st.button("Predict"):
        ouput, probs = predict(raw_test)

        col1,col2 = st.columns(2)

        with col1 :
            st.success("Your Input Text :")
            st.write(raw_test)

            st.success("Predicted Ouput:")
            st.write(ouput)

            st.success("Confidence Score:")
            st.write(np.max(probs))
        
        with col2 :
            columns= ["Sad", "Joy", "Love", "Anger", "Fear", "Surprise"]
            fig = plt.figure()
            plt.style.use("https://github.com/dhaitz/matplotlib-stylesheets/raw/master/pitayasmoothie-dark.mplstyle")
            plt.bar(columns,probs[0])
            st.write(fig)

if __name__ == '__main__':
    main()