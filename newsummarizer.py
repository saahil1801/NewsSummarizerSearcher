import streamlit as st
import pandas as pd
from newspaper import Article
from transformers import pipeline
from GoogleNews import GoogleNews
import re

# Initialize the summarization pipeline
summarizer = pipeline("summarization")

# Create a function to extract article content
def extract_article_content(link):
    article = Article(link)
    article.download()
    article.parse()
    return article.text

def extract_clean_url(link):
    match = re.search(r'(?<=url=)([^&]+)', link)
    if match:
        return match.group(1)
    return link

def extract_article_summary(link):

    article = Article(link)
    article.download()
    article.parse()
    article.nlp()
    
    return article.summary

# Create a Streamlit app
# Create a Streamlit app
def main():
    st.title("News Search & Summarizer Tool")

    # User input: Search query
    search_query = st.text_input("Enter a search query:", "")

    if search_query:
        # Perform search using GoogleNews
        googlenews = GoogleNews(lang="en", period="7d")
        googlenews.search(search_query)

        # Get the search results
        top_news = googlenews.results()
        articles_data = [
            {"Title": news["title"], "Link": news["link"]} for news in top_news
        ]
        articles_df = pd.DataFrame(articles_data)
        articles_df['Cleaned Link'] = articles_df['Link'].apply(extract_clean_url)  # Corrected column name
        articles_df = articles_df[~articles_df['Cleaned Link'].str.contains('youtube.com')]

        # Display search results
        st.subheader("Search Results:")
        for index, row in articles_df.iterrows():
            st.write(f"{index+1}. [{row['Title']}]({row['Cleaned Link']})")

        selected_article = st.selectbox("Select an article:", articles_df["Title"])

        # Extract and display article content
        article_link = articles_df.loc[articles_df["Title"] == selected_article, "Cleaned Link"].iloc[0]
        article_content = extract_article_summary(article_link)
        
        st.subheader("Article Short Summary:")
        st.write(article_content)

        # Summarize the article content
        summary = summarizer(article_content, max_length=50, min_length=30, do_sample=False)
        summarized_text = summary[0]["summary_text"]

        st.subheader("Article Shorter Summary:")
        st.write(summarized_text)

if __name__ == "__main__":
    main()
