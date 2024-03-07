'''
1. Import necessary libraries:
- import pandas as pd
- import spacy
- from textblob import TextBlob

2. Load the spaCy model:
- nlp = spacy.load("en_core_web_sm")

3. Define a function for sentiment analysis of testing statements:
- Function name: analyze_sentiment_testing
- Input: review (str)
- Process:
- Create a TextBlob object for the testing statement
- Determine the polarity score of the statement using TextBlob
- Classify sentiment based on the polarity score
- Output: sentiment (str) - positive, negative, or neutral

4. Define a function for sentiment analysis of dataset reviews:
- Function name: analyze_sentiment_dataset
- Input: review (str)
- Process:
- Create a TextBlob object for the review
- Determine the polarity score of the review using TextBlob
- Classify sentiment based on the polarity score
- Output: sentiment (str) - positive, negative, or neutral

5. Load the dataset:
- Read CSV file into DataFrame using pandas

6. Apply sentiment analysis to dataset reviews:
- Apply analyze_sentiment_dataset function to 'reviews.text' column
- Store the results in a new column named 'sentiment_dataset'

7. Display sentiment analysis results for dataset reviews:
- Print the first few rows of the DataFrame with 'reviews.text' and 'sentiment_dataset' columns

8. Define testing statements:
- Create a list of testing statements

9. Analyze sentiment for each testing statement:
- Iterate over testing statements
- Apply analyze_sentiment_testing function to each statement
- Print sentiment analysis results for each testing statement
'''

import pandas as pd
import spacy
from textblob import TextBlob
from spacy.lang.en.stop_words import STOP_WORDS

# Load the spaCy model
nlp = spacy.load("en_core_web_sm")

# Function for sentiment analysis of testing statements
def analyze_sentiment_testing(review):
    """
    Analyze the sentiment of a testing statement using TextBlob.

    Arguments:
    - review (str): The testing statement text.

    Returns:
    - sentiment (str): The sentiment of the statement ('positive', 'negative', or 'neutral').
    """
    # Perform sentiment analysis using TextBlob
    blob = TextBlob(review)
    polarity = blob.sentiment.polarity

    # Classify sentiment based on polarity score
    if polarity > 0:
        sentiment = 'positive'
    elif polarity < 0:
        sentiment = 'negative'
    else:
        sentiment = 'neutral'
    
    return sentiment

# Function for sentiment analysis of dataset reviews
def analyze_sentiment_dataset(review):
    """
    Analyze the sentiment of a product review using TextBlob.

    Argumentss:
    - review (str): The product review text.

    Returns:
    - sentiment (str): The sentiment of the review ('positive', 'negative', or 'neutral').
    """
    # Performing sentiment analysis using TextBlob
    blob = TextBlob(review)
    polarity = blob.sentiment.polarity

    # Classifying sentiment based on polarity score
    if polarity > 0:
        sentiment = 'positive'
    elif polarity < 0:
        sentiment = 'negative'
    else:
        sentiment = 'neutral'
    
    return sentiment

# Function for sentiment analysis of the entire dataset
def analyze_sentiment(review):
    """
    Analyse the sentiment of a review using TextBlob.

    Argumentss:
    - review (str): The review text.

    Returns:
    - sentiment (str): The sentiment of the review ('positive', 'negative', or 'neutral').
    """
    # Performing sentiment analysis using TextBlob
    blob = TextBlob(review)
    polarity = blob.sentiment.polarity

    # Classifying sentiment based on polarity score
    if polarity > 0:
        sentiment = 'positive'
    elif polarity < 0:
        sentiment = 'negative'
    else:
        sentiment = 'neutral'
    
    return sentiment

# Load the dataset
df = pd.read_csv("/Users/jamespritchard/Desktop/Programming/Bootcamp/Capstone/Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products_May19.csv")

# Applying the  sentiment analysis to dataset reviews
df['sentiment_dataset'] = df['reviews.text'].apply(analyze_sentiment_dataset)

# Define function to clean text
def clean_text(text):
    doc = nlp(text)
    tokens = [token.text.lower() for token in doc if not token.is_stop and not token.is_punct]
    return " ".join(tokens)

# Applying text cleaning to dataset reviews
df['clean_reviews'] = df['reviews.text'].apply(clean_text)

# Displaying the first few rows of the DataFrame with the cleaned text column added
print("Cleaned text:")
print(df[['reviews.text', 'clean_reviews']].head())

# Example reviews for testing
print("\nSentiment analysis for testing statements:")
reviews_testing = [
    "This product is amazing! I absolutly love it.",
    "The quality of this product is tremendously bad. I am very disappointed.",
    "I'm satisfied with this particular purchase.",
    "The customer service was atrocious. I will never buy from this brand ever again.",
    "I am 100 percent in love with this product."
]

# Analysing the sentiment for the testing statements
for i, review in enumerate(reviews_testing, start=1):
    sentiment = analyze_sentiment_testing(review)
    print(f"Sentiment for Testing Statement {i}: {sentiment}")

# Applying the sentiment analysis to the entire dataset
df['sentiment'] = df['clean_reviews'].apply(analyze_sentiment)

# Displaying the first few rows of the DataFrame with the sentiment column added
print("\nSentiment analysis for dataset reviews (applied to entire dataset):")
print(df[['clean_reviews', 'sentiment']].head())