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
import matplotlib.pyplot as plt
from PIL import Image

# Data dependencies
import pandas as pd

import nltk
nltk.download('wordnet')
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
test = pd.read_csv("Data/test.csv")

pattern_url = r'http[s]?://(?:[A-Za-z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9A-Fa-f][0-9A-Fa-f]))+'
subs_url = r'url-web'
train['message'] = train['message'] .replace(to_replace = pattern_url, value = subs_url, regex = True)
test['message'] = test['message'] .replace(to_replace = pattern_url, value = subs_url, regex = True)

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

test['Processed_message'] = test.message.apply(lambda x: clean_text(x))



# The main function where we will build the actual app
def main():
	
	html_temp = """<div style="background-color:tansparent;"><div class="header-category__background" style="background-image: url('https://img.freepik.com/free-photo/pile-3d-twitter-logos_1379-879.jpg?size=620&ext=jpg');"><p style="color:white;font-size:50px;padding:50px">TWEET CLASSIFIER</p></div>"""
	st.markdown(html_temp,unsafe_allow_html=True)
	# Creates a main title and subheader on your page -
	# these are static across all pagesss
	image = Image.open('_110627626_trump_climate_quotesv7_976-nc.png').convert('RGB')
	st.image(image, caption='TRUMP TWEET QOUTES', use_column_width=True)
	# Creating sidebar with selection box -
	# you can create multiple pages this way
	st.subheader("Climate change tweet classification")
	options = ["Prediction", "Information", "Models", "EDA"]
	selection = st.sidebar.selectbox("Choose Option", options)
	if selection == "Models":
		st.info('The infomation about the models')
		# You can read a markdown file from supporting resources folder
		html = markdown.markdown(open("Data/models.md").read())
		st.markdown(html, unsafe_allow_html=True )

	# Building out the "Information" page
	if selection == "Information":
		st.info("General Information")
		# You can read a markdown file from supporting resources folder
		html = markdown.markdown(open("Data/info.md").read())
		st.markdown(html, unsafe_allow_html=True)

		st.subheader("Raw Twitter data and label")
		if st.checkbox('Show raw data'): # data is hidden if box is unchecked
			st.write(train[['sentiment', 'message']]) # will write the df to the page

	if selection =="EDA":
		st.subheader("The Visualizations used to explore the raw and processed tweeter data")
		if st.checkbox('The popular words used in the Tweets message data'): # data is hidden if box is unchecked
			image = Image.open('joint_cloud.png')
			st.image(image, caption='WORD CLOUD ', use_column_width=True)
		if st.checkbox('Tweet message distribution over the sentiments pie chart'): # data is hidden if box is unchecked
			image = Image.open('Tweet message distribution over the sentiments bar chart.png')
			st.image(image, caption='Tweet message distribution over the sentiments bar chart', use_column_width=True)
		if st.checkbox('Tweet message distribution over the sentiments bar chart'): # data is hidden if box is unchecked
			image = Image.open('Tweet message distribution over the sentiments.png')
			st.image(image, caption='Tweet message distribution over the sentiments ', use_column_width=True)
		if st.checkbox('The count of word used in the Tweets message data'): # data is hidden if box is unchecked
			image = Image.open('wordcount_bar.png')
			st.image(image, caption='WORD COUNT BAR', use_column_width=True)
		
	
	# Building out the predication page
	if selection == "Prediction":
		st.info("Prediction with ML Models")
		# Creating a text box for user input
		tweet_text = st.text_area("Enter Text/Tweet","Type Here")
		st.subheader("Select a check box of the model you wish to use to classify your tweet")
		if st.checkbox("LinearSVC"):
			# Transforming user input with vectorizer
			vect_text = vectoriser.transform([tweet_text]).toarray()
			# Load your .pkl file with the model of your choice + make predictions
			# Try loading in multiple models to give the user a choice
			predictor = joblib.load(open(os.path.join("Data/LinearSVC.pkl"),"rb"))
			prediction = predictor.predict(vect_text)
			st.success("Text Categorized as: {}".format(prediction))
			st.balloons()
		if st.checkbox('Logistic'):
			# Transforming user input with vectorizer
			vect_text = vectoriser.transform([tweet_text]).toarray()
			# Load your .pkl file with the model of your choice + make predictions
			# Try loading in multiple models to give the user a choice
			predictor = joblib.load(open(os.path.join("Data/LogisticRegression.pkl"),"rb"))
			prediction = predictor.predict(vect_text)
			st.success("Text Categorized as: {}".format(prediction))
			st.balloons()
	
		if st.checkbox('SVC'):
			# Transforming user input with vectorizer
			vect_text = vectoriser.transform([tweet_text]).toarray()
			# Load your .pkl file with the model of your choice + make predictions
			# Try loading in multiple models to give the user a choice
			predictor = joblib.load(open(os.path.join("Data/SVC.pkl"),"rb"))
			prediction = predictor.predict(vect_text)
			st.success("Text Categorized as: {}".format(prediction))
			st.balloons()
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
			st.balloons()

# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()
