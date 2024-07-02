import streamlit as st
from textblob import TextBlob
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import cleantext
import matplotlib.pyplot as plt

# Getting dataset
data = pd.read_csv('Reviews.csv')

# Initialize the VADER sentiment intensity analyzer
analyzer = SentimentIntensityAnalyzer()
review_text = data['Text']

# Analyze sentiment and subjectivity
sentiment_scores = []
blob_subj = []
for reviews in review_text:
    sentiment_scores.append(analyzer.polarity_scores(reviews)['compound'])
    blob = TextBlob(reviews)
    blob_subj.append(blob.subjectivity)

# Classify sentiment based on the VADER scores
sentiment_classes = []
for score in sentiment_scores:
    if score > 0.8:
        sentiment_classes.append('Highly positive')
    elif score > 0.4:
        sentiment_classes.append('positive')
    elif -0.4 <= score <= 0.4:
        sentiment_classes.append('neutral')
    elif score <= -0.4:
        sentiment_classes.append('negative')
    else:
        sentiment_classes.append('Highly negative')

# Streamlit
st.title("Sentiment Analysis on Customer Feedback")

# User input
user_input = st.text_area('Enter the feedback: ')
blob = TextBlob(user_input)

user_sentiment_score = analyzer.polarity_scores(user_input)['compound']

if user_sentiment_score > 0.8:
    user_sentiment_class = 'Highly positive'
elif user_sentiment_score > 0.4:
    user_sentiment_class = 'positive'
elif -0.4 <= user_sentiment_score <= 0.4:
    user_sentiment_class = 'neutral'
elif user_sentiment_score <= -0.4:
    user_sentiment_class = 'negative'
else:
    user_sentiment_class = 'Highly negative'

st.write('VADER Sentiment Class: **', user_sentiment_class, '** Vader Sentiment Score: **', user_sentiment_score)
st.write('**TextBlob Sentiment Class: **', blob.sentiment.polarity, '** TextBlob Subjectivity: **', blob.sentiment.subjectivity)

# Display clean text
pre = st.text_input('Clean Text')
if pre:
    st.write(cleantext.clean(pre, clean_all=False, extra_spaces=True, stopwords=True, lowercase=True, numbers=True, punct=True))
else:
    st.write('No text has been provided by the user for cleaning')

# Graphical representation of the data
st.subheader("Graphical Representation of Data")
plt.figure(figsize=(10,6))

sentiment_scores_by_class = {k: [] for k in set(sentiment_classes)}
for score, sentiment_class in zip(sentiment_scores, sentiment_classes):
    sentiment_scores_by_class[sentiment_class].append(score)

for sentiment_class, scores in sentiment_scores_by_class.items():
    plt.hist(scores, label=sentiment_class, alpha=0.50)

plt.xlabel("Sentiment Score")
plt.ylabel('Count')
plt.title('Score Distribution')
plt.legend()
st.pyplot(plt)

# Dataframes with the sentiment analysis results
data['Sentiment Class'] = sentiment_classes
data['Sentiment Score'] = sentiment_scores
data['Subjectivity'] = blob_subj

new_data = data[["Score", "Sentiment Score", "Sentiment Class", "Subjectivity"]]
st.subheader("Input Dataframe")
st.dataframe(new_data.head(10), use_container_width=True)
