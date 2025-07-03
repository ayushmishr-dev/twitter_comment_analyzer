import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from apify_client import ApifyClient
import requests
import re
import os
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from dotenv import load_dotenv

st.set_page_config(page_title="Twitter Comment Analyzer", layout="wide")
st.title("🐦 Twitter Post Comment Analyzer (Sentiment + Topic Modeling)")

load_dotenv()
APIFY_TOKEN = os.getenv("APIFY_TOKEN")

def extract_comment(item):
    for field in ['text', 'content', 'replyText']:
        if field in item and item[field]:
            return item[field]
    return None

def perform_sentiment_analysis(df):
    analyzer = SentimentIntensityAnalyzer()
    pos_comments, neg_comments, neu_comments = [], [], []

    for comment in df['Comment']:
        score = analyzer.polarity_scores(comment)['compound']
        if score >= 0.05:
            pos_comments.append(comment)
            neg_comments.append('')
            neu_comments.append('')
        elif score <= -0.05:
            pos_comments.append('')
            neg_comments.append(comment)
            neu_comments.append('')
        else:
            pos_comments.append('')
            neg_comments.append('')
            neu_comments.append(comment)

    sentiment_df = pd.DataFrame({
        "Positive Comments": pos_comments,
        "Negative Comments": neg_comments,
        "Neutral Comments": neu_comments
    })

    sentiment_df.to_csv("sentiment_analysis.csv", index=False)

    sentiment_counts = {
        'Positive': sum([1 for c in pos_comments if c]),
        'Negative': sum([1 for c in neg_comments if c]),
        'Neutral': sum([1 for c in neu_comments if c])
    }

    return sentiment_df, sentiment_counts

def clean_text(text):
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text.lower()

def perform_topic_modeling(df, num_topics=3):
    stop_words = stopwords.words('english')
    texts = df['Comment'].apply(clean_text)

    vectorizer = CountVectorizer(stop_words=stop_words, max_features=1000)
    doc_term_matrix = vectorizer.fit_transform(texts)

    lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    lda.fit(doc_term_matrix)

    topic_keywords = []
    for topic in lda.components_:
        words = [vectorizer.get_feature_names_out()[i] for i in topic.argsort()[-3:][::-1]]
        topic_keywords.append(", ".join(words))

    doc_topics = lda.transform(doc_term_matrix)
    assigned_topics = doc_topics.argmax(axis=1)
    comment_topics = [topic_keywords[i] for i in assigned_topics]

    topic_df = pd.DataFrame({
        "Comment": df['Comment'],
        "Theme/Keywords": comment_topics
    })

    topic_df.to_csv("topic_modeling.csv", index=False)
    return topic_df

def extract_comments(tweet_url):
    client = ApifyClient(APIFY_TOKEN)
    run_input = {
        "postUrls": [tweet_url],
        "resultsLimit": 50,
        "includeNestedReplies": False,
        "repliesSortType": "relevancy",
        "proxyConfiguration": {"useApifyProxy": True}
    }

    try:
        run = client.actor("scraper_one/x-post-replies-scraper").call(run_input=run_input)
        dataset_id = run["defaultDatasetId"]

        dataset_url = f"https://api.apify.com/v2/datasets/{dataset_id}/items?token={APIFY_TOKEN}"
        response = requests.get(dataset_url)

        if response.status_code != 200:
            st.error("Failed to fetch dataset.")
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {}

        data = response.json()
        comments = [{"Comment": extract_comment(item)} for item in data if extract_comment(item)]
        if not comments:
            st.warning("No valid comments extracted.")
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {}

        df = pd.DataFrame(comments)
        df.to_csv("comments.csv", index=False)

        sentiment_df, sentiment_counts = perform_sentiment_analysis(df)
        topic_df = perform_topic_modeling(df)

        return df, sentiment_df, topic_df, sentiment_counts

    except Exception as e:
        st.error(f"Error: {e}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {}

tweet_url = st.text_input("Paste Twitter Post URL:", "")

if st.button("Analyze"):
    if tweet_url.strip() == "" or "x.com" not in tweet_url:
        st.warning("Please enter a valid X/Twitter post URL.")
    else:
        with st.spinner("Extracting and analyzing comments..."):
            df, sentiment_df, topic_df, sentiment_counts = extract_comments(tweet_url)

            if not df.empty:
                st.success("✅ Comments extracted.")
                st.subheader("🔹 Sample Comments")
                st.dataframe(df.head())

                st.subheader("🔹 Sentiment Analysis")
                st.dataframe(sentiment_df.head())

                # Sentiment Visualization
                st.subheader("📊 Sentiment Breakdown")
                labels = list(sentiment_counts.keys())
                sizes = list(sentiment_counts.values())
                colors = ['#90EE90', '#FFCCCB', '#D3D3D3']

                fig, ax = plt.subplots()
                ax.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=140)
                ax.axis('equal')
                st.pyplot(fig)

                st.subheader("🔹 Topic Modeling")
                st.dataframe(topic_df.head())

                st.download_button("Download comments.csv", df.to_csv(index=False), "comments.csv", "text/csv")
                st.download_button("Download sentiment_analysis.csv", sentiment_df.to_csv(index=False), "sentiment_analysis.csv", "text/csv")
                st.download_button("Download topic_modeling.csv", topic_df.to_csv(index=False), "topic_modeling.csv", "text/csv")
