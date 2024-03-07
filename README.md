				## Sentiment Analysis Report 
												
1. [Introduction]
2. [Methodology]
   3. [Preprocessing Steps]
   4. [Sentiment Analysis Approach]
5. [Implementation]
   6. [Importing Necessary Libraries]
   7. [Loading the Dataset]
   8. [Functions for Sentiment Analysis]
9. [Results and Evaluation]
   10. [Dataset Sentiment Analysis]
   11. [Testing Statements Sentiment Analysis]
12. [Insights and Recommendations]
   13. [Model's Strengths]
   14. [Model's Limitations]
   15. [Recommendations for Improvement]
16. [Conclusion]

    
## 1. Introduction

### 1.1 Background
   - Sentiment analysis, also known as opinion mining, is the process of analysing text to determine the sentiment expressed within it. It is widely used in various fields, including marketing, customer feedback analysis, and social media monitoring, to understand public opinion and sentiment towards products, services, or topics.
   
### 1.2 Dataset Description
   - The dataset used for sentiment analysis consists of Amazon consumer reviews of Amazon products collected in May 2019. It contains a large number of reviews across different product categories, providing a diverse set of opinions and sentiments to analyse.

## 2. Methodology

### 2.1 Preprocessing Steps
   - Prior to sentiment analysis, the dataset underwent preprocessing to clean and prepare the text data. This included:
     - Removal of stop words and punctuation using spaCy.
     - Tokenisation and lemmatisation to convert words into their base forms.
     - Text cleaning to remove any irrelevant or noisy information.

# Apply sentiment analysis to dataset reviews
df['sentiment_dataset'] = df['reviews.text'].apply(analyze_sentiment_dataset)

# Define function to clean text
def clean_text(text):
    doc = nlp(text)
    tokens = [token.text.lower() for token in doc if not token.is_stop and not token.is_punct]
    return " ".join(tokens)

# Apply text cleaning to dataset reviews
df['clean_reviews'] = df['reviews.text'].apply(clean_text)

# Display the first few rows of the DataFrame with the cleaned text column added
print("Cleaned text:")
print(df[['reviews.text', 'clean_reviews']].head())

### 2.2 Sentiment Analysis Approach
   - Sentiment analysis was performed using the TextBlob library, which provides a simple API for natural language processing tasks such as sentiment analysis. The approach involved:
     - Creating a TextBlob object for each review or testing statement.
     - Calculating the polarity score of the text, representing its sentiment on a scale from -1 (negative) to 1 (positive).
     - Classifying the sentiment based on the polarity score.

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

## 3. Implementation

### 3.1 Importing Necessary Libraries
   - The following Python libraries were imported for sentiment analysis:
     - `pandas` (alias `pd`): Used for data manipulation and analysis.
     - `spacy`: An open-source natural language processing library.
     - `TextBlob`: A Python library for processing textual data.
     - `STOP_WORDS` from `spacy.lang.en.stop_words`: Provides a set of common stop words for English.

import pandas as pd
import spacy
from textblob import TextBlob
from spacy.lang.en.stop_words import STOP_WORDS

### 3.2 Loading the Dataset
   - The dataset was loaded into a DataFrame using the `pd.read_csv()` function from the `pandas` library. This allowed for easy manipulation and analysis of the data.

# Load the dataset
df = pd.read_csv("/Users/jamespritchard/Desktop/Programming/Bootcamp/Capstone/Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products_May19.csv")

### 3.3 Functions for Sentiment Analysis
   #### 3.3.1 Function: `analyze_sentiment_dataset()`
   - This function analyses the sentiment of a product review using TextBlob.
   - Input: `review` (str) - The product review text.
   - Output: `sentiment` (str) - The sentiment of the review ('positive', 'negative', or 'neutral').
   - Process:
     - Create a TextBlob object for the review.
     - Determine the polarity score of the review using TextBlob.
     - Classify sentiment based on the polarity score.

# Function for sentiment analysis of dataset reviews
def analyze_sentiment_dataset(review):
    """
    Analyze the sentiment of a product review using TextBlob.

    Args:
    - review (str): The product review text.

    Returns:
    - sentiment (str): The sentiment of the review ('positive', 'negative', or 'neutral').
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

   #### 3.3.2 Function: `analyze_sentiment_testing()`
   - This function analyses the sentiment of a testing statement using TextBlob.
   - Input: `review` (str) - The testing statement text.
   - Output: `sentiment` (str) - The sentiment of the statement ('positive', 'negative', or 'neutral').
   - Process:
     - Create a TextBlob object for the testing statement.
     - Determine the polarity score of the statement using TextBlob.
     - Classify sentiment based on the polarity score.

# Function for sentiment analysis of testing statements
def analyze_sentiment_testing(review):
    """
    Analyze the sentiment of a testing statement using TextBlob.

    Args:
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

   #### 3.3.3 Function: `analyze_sentiment()`
   - This function analyses the sentiment of a review using TextBlob.
   - Input: `review` (str) - The review text.
   - Output: `sentiment` (str) - The sentiment of the review ('positive', 'negative', or 'neutral').
   - Process:
     - Create a TextBlob object for the review.
     - Determine the polarity score of the review using TextBlob.
     - Classify sentiment based on the polarity score.

# Function for sentiment analysis of entire dataset
def analyze_sentiment(review):
    """
    Analyse the sentiment of a review using TextBlob.

    Args:
    - review (str): The review text.

    Returns:
    - sentiment (str): The sentiment of the review ('positive', 'negative', or 'neutral').
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

### 3.4 Applying Sentiment Analysis
   - Sentiment analysis was applied to both the dataset reviews and testing statements using the defined functions. The results were stored in new columns in the DataFrame for further analysis and visualisation.

## 4. Results and Evaluation

### 4.1 Dataset Sentiment Analysis
   - The sentiment analysis results for the dataset reviews were evaluated based on their accuracy in capturing the sentiment expressed in the reviews. The distribution of sentiments across different product categories was also analysed to identify trends and patterns.

### 4.2 Testing Statements Sentiment Analysis
   - The sentiment analysis results for the testing statements were evaluated to assess the model's performance in classifying sentiment in different contexts. Any discrepancies or misclassifications were noted and analysed to identify areas for improvement.

## 5. Insights and Recommendations

### 5.1 Model's Strengths
   - The model demonstrated strengths in accurately capturing sentiment across a wide range of reviews and testing statements.
   - It provided valuable insights into customer sentiment towards various Amazon products.

### 5.2 Model's Limitations
   - The model may struggle with sarcasm, irony, or nuanced language, leading to misclassifications in sentiment analysis.
   - It relies on predefined sentiment analysis libraries, which may not capture all contextualised nuances or cultural differences in languages.

### 5.3 Recommendations for Improvement
   - To improve the accuracy of sentiment analysis, further fine-tuning of the model may be necessary.
   - Incorporating additional features or data sources, such as user demographics or product attributes.

## 6. Conclusion
   - In conclusion, the sentiment analysis of Amazon consumer reviews using TextBlob provided valuable insights into customer sentiment towards Amazon products. While the model demonstrated strengths in capturing sentiment, there are opportunities for improvement to enhance its accuracy and robustness in analysing sentiment in diverse contexts.
