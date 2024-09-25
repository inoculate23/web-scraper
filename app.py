import streamlit as st
import requests
from youtube_transcript_api import YouTubeTranscriptApi
from bs4 import BeautifulSoup
import re
import urllib.parse as urlparse
from fake_useragent import UserAgent
import cloudscraper
import time
import random
import pandas as pd
import base64
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import google.generativeai as genai
import tiktoken
import json
import os
from dotenv import load_dotenv

load_dotenv()

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

st.set_page_config(layout="wide", page_title="Web Scraper", page_icon="🔍")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap');
body {
    font-family: 'Poppins', sans-serif;
    background-color: #f0f4f8;
    color: #1a202c;
}
.big-font {
    font-size: 48px !important;
    font-weight: 600;
    color: #2d3748;
    text-align: center;
    margin-bottom: 20px;
    animation: fadeIn 1.5s ease-out;
}
.medium-font {
    font-size: 24px !important;
    color: #4a5568;
    text-align: center;
    margin-bottom: 30px;
}
.stButton>button {
    background-color: #4299e1;
    color: white;
    font-weight: bold;
    border-radius: 30px;
    padding: 10px 20px;
    border: none;
    box-shadow: 0 4px 6px rgba(66, 153, 225, 0.3);
    transition: all 0.3s;
}
.stButton>button:hover {
    background-color: #3182ce;
    box-shadow: 0 6px 8px rgba(49, 130, 206, 0.4);
    transform: translateY(-2px);
}
.stTextInput>div>div>input {
    border-radius: 30px;
    border: 2px solid #4299e1;
    padding: 10px 20px;
    transition: all 0.3s;
}
.stTextInput>div>div>input:focus {
    border-color: #3182ce;
    box-shadow: 0 0 0 2px rgba(49, 130, 206, 0.2);
    transform: scale(1.02);
}
.summary-box {
    background-color: #ebf8ff;
    border-radius: 10px;
    padding: 20px;
    margin-top: 20px;
    box-shadow: 0 4px 6px rgba(66, 153, 225, 0.1);
    transition: all 0.3s;
}
.summary-box:hover {
    box-shadow: 0 6px 8px rgba(66, 153, 225, 0.2);
    transform: translateY(-2px);
}
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(-20px); }
    to { opacity: 1; transform: translateY(0); }
}
@keyframes float {
    0% { transform: translateY(0px); }
    50% { transform: translateY(-10px); }
    100% { transform: translateY(0px); }
}
.magnify {
    display: inline-block;
    animation: float 3s ease-in-out infinite;
    margin-right: 10px;
}
.tab-content {
    animation: fadeIn 0.5s ease-out;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="big-font"><span class="magnify">🔍</span>Web Scraper</p>', unsafe_allow_html=True)
st.markdown('<p class="medium-font">Extract, summarize and analyze content from any website or YouTube video with ease!</p>', unsafe_allow_html=True)

url_input = st.text_input("Enter URL to scrape (Website or YouTube)", key="url_input")
scrape_button = st.button("🚀 Start Scraping")

def get_human_like_user_agent():
    ua = UserAgent()
    return ua.random

def extract_video_id(url):
    parsed_url = urlparse.urlparse(url)
    if parsed_url.hostname == 'youtu.be':
        return parsed_url.path[1:]
    if parsed_url.hostname in ('www.youtube.com', 'youtube.com'):
        if parsed_url.path == '/watch':
            query = urlparse.parse_qs(parsed_url.query)
            return query['v'][0]
        if parsed_url.path[:7] == '/embed/':
            return parsed_url.path.split('/')[2]
        if parsed_url.path[:3] == '/v/':
            return parsed_url.path.split('/')[2]
    raise ValueError("Invalid YouTube URL")

def fetch_youtube_transcript(video_id):
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        transcript_text = ' '.join([x['text'] for x in transcript])
        return transcript_text
    except Exception as e:
        st.error(f"Error fetching YouTube transcript: {str(e)}")
        return None

def scrape_website_text(url):
    headers = {'User-Agent': get_human_like_user_agent()}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            text = extract_text_from_soup(soup)
            if text:
                return text

        scraper = cloudscraper.create_scraper()
        response = scraper.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            text = extract_text_from_soup(soup)
            if text:
                return text

        return None
    except Exception as e:
        st.error(f"Error scraping website: {str(e)}")
        return None

def extract_text_from_soup(soup):
    for script in soup(["script", "style"]):
        script.decompose()
    text = ' '.join(soup.stripped_strings)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def main_scraper(url):
    if 'youtube.com' in url or 'youtu.be' in url:
        video_id = extract_video_id(url)
        text = fetch_youtube_transcript(video_id)
        source_type = "YouTube Transcript"
    else:
        text = scrape_website_text(url)
        source_type = "Website Content"

    return text, source_type

def estimate_tokens(text):
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    tokens = encoding.encode(text)
    return len(tokens)

def get_content_summary(text):
    sentences = nltk.sent_tokenize(text)
    words = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word.isalnum() and word not in stop_words]

    return {
        'estimated_tokens': estimate_tokens(text),
        'word_count': len(words),
        'sentence_count': len(sentences),
        'avg_words_per_sentence': len(words) / len(sentences) if sentences else 0,
        'unique_words': len(set(words))
    }

def generate_ai_summary(text):
    api_key = st.secrets["GEMINI_API_KEY"]

    genai.configure(api_key=api_key)

    generation_config = {
      "temperature": 1,
      "top_p": 0.95,
      "top_k": 40,
      "max_output_tokens": 8192,
    }

    model = genai.GenerativeModel(
      model_name="gemini-1.5-flash-002",
      generation_config=generation_config,
    )

    prompt = """
    You are tasked with summarizing content from any source (e.g., research papers, YouTube transcripts, blogs, stories, or any other text) in a concise, yet comprehensive manner. Your goal is to present the summary in a way that captures all the important information without missing key details or paraphrasing too much. The summary should be easily understood and structured in a way that reflects the original content's form, tone, and complexity. Adjust your level of explanation dynamically, simplifying only where necessary to aid understanding without losing the original meaning or technical depth. Follow these instructions carefully:

    1. **Identify the content type and adapt your style accordingly**:
       - Recognize if the input is a research paper, blog post, YouTube transcript, story, or any other type of content. Begin the summary by acknowledging this context and ensuring the narration matches the content type:
         - For formal content (e.g., research papers, reports), maintain a professional tone, focusing on key findings, methods, and conclusions.
         - For informal content (e.g., blogs, YouTube transcripts), maintain a conversational tone while preserving the main points and the author's or speaker's intent.

    2. **Use third-person narration**:
       - Always refer to the author or speaker in third person. Describe what they are doing, saying, or arguing.
         - For example: "The author discusses...", "The speaker introduces...", "The presenter highlights..."
       - This helps the user understand the source and the flow of ideas.

    3. **Break down content into sections and key points**:
       - Divide the summary into sections that reflect the original structure of the content, using bullet points where necessary.
       - Ensure each section summarizes the main points clearly, while capturing the depth of more complex sections.
         - For straightforward sections, keep the summary concise.
         - For complex or technical sections, provide more explanation to ensure clarity.

    4. **Dynamically adjust for complex topics**:
       - When encountering complex, technical, or difficult concepts, adjust your summarization by explaining these topics in simpler terms where necessary. However, **do not lose technical accuracy** or **omit important details**.
         - Use explanations that help the user **understand the underlying concepts** without removing critical details.
         - Example: *If the content involves a technical process, provide a simplified explanation of how it works, but also retain the key technical steps or terminology as used in the original.*

    5. **Capture all important information without paraphrasing too much**:
       - Ensure that the summary doesn't oversimplify or skip key ideas, particularly in technical or highly detailed sections.
       - Preserve critical information such as key findings, conclusions, and supporting arguments in a balanced manner.
       - Example: *If the content presents technical details, keep the terminology intact and briefly explain complex concepts when necessary.*

    6. **Adapt the level of detail based on the content's complexity**:
       - Provide high-level summaries for general or repetitive points, but expand on more complex sections (e.g., technical arguments, controversial points, detailed explanations).
       - Example: *For sections that involve difficult-to-understand content, break down the information into digestible steps or explanations, helping the reader comprehend the core ideas while preserving technical precision.*

    7. **Preserve the tone and logical flow**:
       - Ensure the summary follows the logical progression of the content, showing how one point leads to the next.
       - Maintain the original tone—whether formal, informal, optimistic, or neutral—without introducing external bias or personal interpretation.

    8. **Teach the user where necessary**:
       - If the content presents especially complex or challenging topics, aim to explain these topics in a way that educates the user, similar to how a human would break down difficult concepts for easier understanding.
       - Example: *When summarizing complex concepts or findings, take extra care to clarify the reasoning or technical process behind them without simplifying the explanation too much.*

    Important: Do not use codeblocks for normal text. Only use codeblocks if there's actual code in the content being summarized.

    Below is the Content/Raw Text extracted from the source:
    {text}

    Please provide the summary based on these specifically crafted instructions.
    """

    response = model.generate_content(prompt.format(text=text))

    if response.parts:
        summary_text = response.parts[0].text
        return summary_text
    else:
        return "Unable to generate summary."

def extract_links(text):
    url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    return url_pattern.findall(text)

def get_json_download_link(data):
    json_str = json.dumps(data, indent=2)
    b64 = base64.b64encode(json_str.encode()).decode()
    href = f'<a href="data:application/json;base64,{b64}" download="scraped_data.json" style="text-decoration:none;"><button style="background-color:#333;color:white;font-weight:bold;border-radius:30px;padding:10px 20px;border:none;box-shadow:0 4px 6px rgba(0,0,0,0.1);transition:all 0.3s;">📥 Download JSON</button></a>'
    return href

if scrape_button and url_input.strip():
    with st.spinner("🔍 Scraping in progress... Please wait."):
        text, source_type = main_scraper(url_input)

        if text:
            st.success("✅ Scraping completed successfully!")

            tab1, tab2, tab3 = st.tabs(["Scraped Raw Text", "AI Summary", "Statistics"])

            with tab1:
                st.markdown('<div class="tab-content">', unsafe_allow_html=True)
                st.subheader(f"📄 Extracted {source_type}")
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("📋 Copy to Clipboard"):
                        st.session_state['clipboard'] = text
                        st.success("✅ Text copied to clipboard!")
                with col2:
                    json_data = {"url": url_input, "content": text}
                    st.markdown(get_json_download_link(json_data), unsafe_allow_html=True)
                st.code(text, language="text")
                st.markdown('</div>', unsafe_allow_html=True)

            with tab2:
                st.markdown('<div class="tab-content">', unsafe_allow_html=True)
                summary_placeholder = st.empty()
                summary_placeholder.info("🤖 AI is generating the summary...")
                try:
                    ai_summary_text = generate_ai_summary(text)
                    summary_placeholder.empty()
                    st.markdown('<div class="summary-box">', unsafe_allow_html=True)
                    st.subheader("🤖 AI Summary")
                    st.write(ai_summary_text)
                    st.markdown('</div>', unsafe_allow_html=True)
                    st.toast("AI Summary generated successfully!", icon="✅")
                except Exception as e:
                    summary_placeholder.error(f"Error generating summary: {str(e)}")
                st.markdown('</div>', unsafe_allow_html=True)

            with tab3:
                st.markdown('<div class="tab-content">', unsafe_allow_html=True)
                st.subheader("📊 Content Analysis")
                summary_stats = get_content_summary(text)

                col1, col2, col3, col4, col5 = st.columns(5)
                with col1:
                    st.metric("🔢 Tokens", summary_stats['estimated_tokens'], delta=None, delta_color="normal")
                with col2:
                    st.metric("📚 Words", summary_stats['word_count'], delta=None, delta_color="normal")
                with col3:
                    st.metric("🆕 Unique Words", summary_stats['unique_words'], delta=None, delta_color="normal")
                with col4:
                    st.metric("📜 Sentences", summary_stats['sentence_count'], delta=None, delta_color="normal")
                with col5:
                    st.metric("📏 Avg. Words/Sentence", f"{summary_stats['avg_words_per_sentence']:.2f}", delta=None, delta_color="normal")

                st.subheader("🔗 Extracted Links")
                links = extract_links(text)
                if links:
                    for link in links:
                        st.markdown(f"<a href='{link}' target='_blank'>{link}</a>", unsafe_allow_html=True)
                else:
                    st.write("No links found in the extracted text.")
                st.markdown('</div>', unsafe_allow_html=True)

        else:
            st.error("Failed to extract content from the provided URL.")

st.markdown("---")
st.markdown("<div style='text-align: center; color: #888;'>Made by <a href='https://sidfeels.netlify.app/' target='_blank'>sidfeels</a> | <a href='https://github.com/sidfeels/web-scraper' target='_blank'>Open Source</a> | <a href='https://buymeacoffee.com/sidfeels' target='_blank'>Support me</a></div>", unsafe_allow_html=True)
