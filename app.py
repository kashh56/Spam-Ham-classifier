import streamlit as st
from ML import spam_ham_checker  # Ensure correct import path

st.title('Spam-Email Classifier')

# Get input from the user
mail_to_check = st.text_area('Enter the email text to classify')

if st.button('Classify'):
    if mail_to_check.strip():
        
        result = spam_ham_checker(mail_to_check.strip())        

        # Display the result to the user
        if result:
            st.subheader(f'This email is classified as: {result}')
        else:
            st.error("Classification failed. Please check your input.")
    else:
        st.warning("Please enter a valid email to check.")
