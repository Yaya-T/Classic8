"""

    Simple Streamlit webserver application for serving developed classification
	models.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Plase follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend the functionality of this script
	as part of your predict project.

	For further help with the Streamlit framework, see:

	https://docs.streamlit.io/en/latest/

"""

# Streamlit dependencies
import streamlit as st
import joblib,os
import pickle
import markdown

# Data dependencies
import pandas as pd

# nltk
from nltk.tokenize import word_tokenize, TreebankWordTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import SnowballStemmer
import nltk
import re

    
# Load the vectoriser.
file = open("Data/vectoriser.pkl","rb")
vectoriser = pickle.load(file)
file.close()
# Load your raw data
train = pd.read_csv("Data/train.csv")

pattern_url = r'http[s]?://(?:[A-Za-z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9A-Fa-f][0-9A-Fa-f]))+'
subs_url = r'url-web'
train['message'] = train['message'] .replace(to_replace = pattern_url, value = subs_url, regex = True)
#test['message'] = test['message'] .replace(to_replace = pattern_url, value = subs_url, regex = True)

stop_words = set(stopwords.words("english")) 
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = re.sub('<[^<]+?>','', text)
    text = re.sub(r'[^\w\s]','',text, re.UNICODE)
    text = text.lower()
    text = [lemmatizer.lemmatize(token) for token in text.split(" ")]
    text = [lemmatizer.lemmatize(token, "v") for token in text]
    text = [word for word in text if not word in stop_words]
    text = " ".join(text)
    return text

train['Processed_message'] = train.message.apply(lambda x: clean_text(x))

#test['Processed_message'] = test.message.apply(lambda x: clean_text(x))



# The main function where we will build the actual app
def main():
	html_temp = """<div style="background-color:purple;"><p style="color:white;font-size:50px;padding:10px">Tweet Classifier App</p></div>"""
	st.markdown(html_temp,unsafe_allow_html=True)
	# Creates a main title and subheader on your page -
	# these are static across all pages
	st.title("Tweet Classifer")
	st.subheader("Climate change tweet classification")

	# Creating sidebar with selection box -
	# you can create multiple pages this way
	options = ["Prediction", "Information", "Models"]
	selection = st.sidebar.selectbox("Choose Option", options)
	if selection == "Models":
		st.info('The infomation about the models')
		# You can read a markdown file from supporting resources folder
		st.markdonw("information abaout the models ")

	# Building out the "Information" page
	if selection == "Information":
		st.info("General Information")
		# You can read a markdown file from supporting resources folder
		st.markdown("Data/info.md")

		st.subheader("Raw Twitter data and label")
		if st.checkbox('Show raw data'): # data is hidden if box is unchecked
			st.write(train[['sentiment', 'message']]) # will write the df to the page

	# Building out the predication page
	if selection == "Prediction":
		st.info("Prediction with ML Models")
		# Creating a text box for user input
		tweet_text = st.text_area("Enter Text/Tweet","Type Here")

	

		if st.checkbox("LinearSVC"):
			# Transforming user input with vectorizer
			vect_text = vectoriser.transform([tweet_text]).toarray()
			# Load your .pkl file with the model of your choice + make predictions
			# Try loading in multiple models to give the user a choice
			predictor = joblib.load(open(os.path.join("Data/LinearSVC.pkl"),"rb"))
			prediction = predictor.predict(vect_text)
			st.success("Text Categorized as: {}".format(prediction))
		if st.checkbox('Logistic'):
			# Transforming user input with vectorizer
			vect_text = vectoriser.transform([tweet_text]).toarray()
			# Load your .pkl file with the model of your choice + make predictions
			# Try loading in multiple models to give the user a choice
			predictor = joblib.load(open(os.path.join("Data/LogisticRegression.pkl"),"rb"))
			prediction = predictor.predict(vect_text)
			st.success("Text Categorized as: {}".format(prediction))
		if st.checkbox('Bernuoli'):
			# Transforming user input with vectorizer
			vect_text = vectoriser.transform([tweet_text]).toarray()
			# Load your .pkl file with the model of your choice + make predictions
			# Try loading in multiple models to give the user a choice
			predictor = joblib.load(open(os.path.join("Data/BNBmodel.pkl"),"rb"))
			prediction = predictor.predict(vect_text)
			st.success("Text Categorized as: {}".format(prediction))
		if st.checkbox('SVC'):
			# Transforming user input with vectorizer
			vect_text = vectoriser.transform([tweet_text]).toarray()
			# Load your .pkl file with the model of your choice + make predictions
			# Try loading in multiple models to give the user a choice
			predictor = joblib.load(open(os.path.join("Data/SVC.pkl"),"rb"))
			prediction = predictor.predict(vect_text)
			st.success("Text Categorized as: {}".format(prediction))
		if st.checkbox('MultiNB'):
			# Transforming user input with vectorizer
			vect_text = vectoriser.transform([tweet_text]).toarray()
			# Load your .pkl file with the model of your choice + make predictions
			# Try loading in multiple models to give the user a choice
			predictor = joblib.load(open(os.path.join("Data/MultinomialNB.pkl"),"rb"))
			prediction = predictor.predict(vect_text)

			# When model has successfully run, will print prediction
			# You can use a dictionary or similar structure to make this output
			# more human interpretable.
			st.success("Text Categorized as: {}".format(prediction))

# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()
