{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Streamlit app"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "# Streamlit dependencies\n",
    "import streamlit as st\n",
    "import joblib,os\n",
    "\n",
    "# Data dependencies\n",
    "import pandas as pd\n",
    "\n",
    "# Vectorizer\n",
    "news_vectorizer = open(\"Data/vectoriser-ngram-(1,2).pkl\",\"wb\")\n",
    "tweet_cv = joblib.load(news_vectorizer) # loading your vectorizer from the pkl file\n",
    "\n",
    "# Load your raw data\n",
    "raw = pd.read_csv(\"Data/train.csv\")\n",
    "\n",
    "# The main function where we will build the actual app\n",
    "\n",
    "def main():\n",
    "    \"\"\"Tweet Classifier App with Streamlit \"\"\"\n",
    "\n",
    "    # Creates a main title and subheader on your page -\n",
    "    # these are static across all pages\n",
    "    st.title(\"Tweet Classifer\")\n",
    "    st.subheader(\"Climate change tweet classification\")\n",
    "\n",
    "    # Creating sidebar with selection box -\n",
    "    # you can create multiple pages this way\n",
    "    options = [\"Prediction\", \"Information\"]\n",
    "    selection = st.sidebar.selectbox(\"Choose Option\", options)\n",
    "\n",
    "    # Building out the \"Information\" page\n",
    "    if selection == \"Information\":\n",
    "        st.info(\"General Information\")\n",
    "        # You can read a markdown file from supporting resources folder\n",
    "        st.markdown(\"Some information here\")\n",
    "\n",
    "        st.subheader(\"Raw Twitter data and label\")\n",
    "        if st.checkbox('Show raw data'): # data is hidden if box is unchecked\n",
    "            st.write(raw[['sentiment', 'message']]) # will write the df to the page\n",
    "\n",
    "    # Building out the predication page\n",
    "    if selection == \"Prediction\":\n",
    "        st.info(\"Prediction with ML Models\")\n",
    "        # Creating a text box for user input\n",
    "        tweet_text = st.text_area(\"Enter Text\",\"Type Here\")\n",
    "\n",
    "        if st.button(\"Classify\"):\n",
    "            # Transforming user input with vectorizer\n",
    "            vect_text = tweet_cv.transform([tweet_text]).toarray()\n",
    "            # Load your .pkl file with the model of your choice + make predictions\n",
    "            # Try loading in multiple models to give the user a choice\n",
    "            predictor = joblib.load(open(os.path.join(\"Data/BNBmodel.pkl\"),\"wb\"))\n",
    "            prediction = predictor.predict(vect_text)\n",
    "\n",
    "            # When model has successfully run, will print prediction\n",
    "            # You can use a dictionary or similar structure to make this output\n",
    "            # more human interpretable.\n",
    "            st.success(\"Text Categorized as: {}\".format(prediction))\n",
    "\n",
    "# Required to let Streamlit instantiate our web app.  \n",
    "if __name__ == '__main__':\n",
    "    main()"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.6"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
