import streamlit as st
import os, re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random 
import io
from pymilvus import MilvusClient

from datetime import datetime, timedelta
from gdeltdoc import Filters, near, repeat, GdeltDoc
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, CacheMode
import asyncio

# from langchain.document_loaders import UnstructuredMarkdownParser
from langchain.schema import Document
from pydantic import BaseModel, Field
from typing import Optional
import configparser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import spacy
from spacytextblob.spacytextblob import SpacyTextBlob
from langchain_community.vectorstores import Milvus

from collections import Counter

# config = configparser.ConfigParser()
# config.read('/content/drive/MyDrive/keys/settings.ini')
# os.environ['DEEPSEEK_KEY'] = config['API_KEYS']['DEEPSEEK_KEY']
# # os.environ['GOOG_KEY'] = config['API_KEYS']['GOOG_AI_STUDIO_KEY']
# os.environ['OPENAI_API_KEY'] = config['API_KEYS']['OPENAI_API_KEY']
os.environ['GOOGLE_API_KEY']= r'AIzaSyAlbksrD6rQFyaf79BgCBrZ5eWGnGiT4as'
os.environ['ZILLIZ_CLOUD_URI']= r'https://in03-8d51ecc9c616de5.serverless.gcp-us-west1.cloud.zilliz.com'
os.environ['ZILLIZ_CLOUD_USERNAME']= r'db_8d51ecc9c616de5'
os.environ['ZILLIZ_CLOUD_PASSWORD']= r'Sj6?Em|q>WLn-y,0'
os.environ['ZILLIZ_CLOUD_API_KEY']= r'5e2dd8ac6c7ff07f12d024e7ea54e16dac4011de21ff7680270d5b0acac07b11e02461544b316dc28d6d51a8b16d1f24d6dc8022'


if "scraped_once" not in st.session_state:
    st.session_state.scraped_once = False

class ArticleSchema(BaseModel):
    main_content: Optional[str] = Field(
        description="Extract only the main body of the article, removing any advertisements, author bios, media links, or unrelated sections."
    )
abv_countries = [["United Arab Emirates", "UAE"],["United Kingdom", "U.K."], ["U.S.","USA", "United States of America"]]
countries = [
    "Afghanistan", "Albania", "Algeria", "Andorra", "Angola",
    "Antigua and Barbuda", "Argentina", "Armenia", "Australia", "Austria",
    "Azerbaijan", "Bahamas", "Bahrain", "Bangladesh", "Barbados",
    "Belarus", "Belgium", "Belize", "Benin", "Bhutan",
    "Bolivia", "Bosnia and Herzegovina", "Botswana", "Brazil", "Brunei",
    "Bulgaria", "Burkina Faso", "Burundi", "Cabo Verde", "Cambodia",
    "Cameroon", "Canada", "Central African Republic", "Chad", "Chile",
    "China", "Colombia", "Comoros", "Congo",
    "Costa Rica", "Croatia", "Cuba", "Cyprus", "Czechia", "Czech Republic",
    "Democratic Republic of the Congo", "Denmark", "Djibouti", "Dominica", "Dominican Republic",
    "Ecuador", "Egypt", "El Salvador", "Equatorial Guinea", "Eritrea",
    "Estonia", "Eswatini", "Ethiopia", "Fiji", "Finland",
    "France", "Gabon", "Gambia", "Georgia", "Germany",
    "Ghana", "Greece", "Grenada", "Guatemala", "Guinea",
    "Guinea-Bissau", "Guyana", "Haiti", "Honduras",
    "Hungary", "Iceland", "India", "Indonesia", "Iran",
    "Iraq", "Ireland", "Israel", "Italy", "Jamaica",
    "Japan", "Jordan", "Kazakhstan", "Kenya", "Kiribati",
    "Kuwait", "Kyrgyzstan", "Laos", "Latvia", "Lebanon",
    "Lesotho", "Liberia", "Libya", "Liechtenstein", "Lithuania",
    "Luxembourg", "Madagascar", "Malawi", "Malaysia", "Maldives",
    "Mali", "Malta", "Marshall Islands", "Mauritania", "Mauritius",
    "Mexico", "Micronesia", "Moldova", "Monaco", "Mongolia",
    "Montenegro", "Morocco", "Mozambique", "Myanmar", "Namibia",
    "Nauru", "Nepal", "Netherlands", "New Zealand", "Nicaragua",
    "Niger", "Nigeria", "North Korea", "North Macedonia", "Norway",
    "Oman", "Pakistan", "Palau", "Palestine State", "Panama",
    "Papua New Guinea", "Paraguay", "Peru", "Philippines", "Poland",
    "Portugal", "Qatar", "Romania", "Russia", "Rwanda",
    "Saint Kitts and Nevis", "Saint Lucia", "Saint Vincent and the Grenadines", "Samoa", "San Marino",
    "Sao Tome and Principe", "Saudi Arabia", "Senegal", "Serbia", "Seychelles",
    "Sierra Leone", "Singapore", "Slovakia", "Slovenia", "Solomon Islands",
    "Somalia", "South Africa", "South Korea", "South Sudan", "Spain",
    "Sri Lanka", "Sudan", "Suriname", "Sweden", "Switzerland",
    "Syria", "Tajikistan", "Tanzania", "Thailand", "Timor-Leste",
    "Togo", "Tonga", "Trinidad and Tobago", "Tunisia", "Turkey",
    "Turkmenistan", "Tuvalu", "Uganda", "Ukraine", "Uruguay", "Uzbekistan", "Vanuatu",
    "Venezuela", "Vietnam", "Yemen", "Zambia", "Zimbabwe"
]


def remove_non_english(text: str) -> str:
    """
    Removes all non-English characters (including digits, punctuation,
    whitespace, etc.) from the given text, leaving only a-z/A-Z.
    """
    return re.sub(r'[^A-Za-z]', '', text)


def search_gdelt_queries(q):
    today_string = datetime.today().strftime('%Y-%m-%d')
    yesterday_string = (datetime.today() - timedelta(days=60)).strftime('%Y-%m-%d')

    # dummy
    f = Filters(
        start_date = yesterday_string,
        end_date = today_string,
        num_records = 250,
        keyword = ['a' ,'b'],
        repeat = repeat(3, "India")
    )
    f.query_params = [q]

    gd = GdeltDoc()
    
    articles_df = gd.article_search(f)
    articles_df = articles_df.drop_duplicates(subset='url')
    articles_df['title_norm'] = articles_df['title'].apply(lambda x: remove_non_english(x.lower()))
    articles_df = articles_df.drop_duplicates(subset= 'title_norm')
    articles_df = articles_df[articles_df['language'] == "English"]
    articles_df = articles_df[~articles_df['url'].str.contains('chinadaily|larouchepub|yahoo|jdsupra|sandiegosun|eurasiareview|insidenova|gdnonline|clutchfans|tomsguide|fool', case=False, na=False)]
    articles_df = articles_df.reset_index(drop=True)

    return articles_df

async def scrape_url(url):
    config = CrawlerRunConfig(
        word_count_threshold=5,
        excluded_tags=['form', 'header', 'footer', 'nav'],
        exclude_social_media_links=True,
        exclude_external_images=True,
        magic=True,
        simulate_user=True,
        override_navigator=True
    )
    try:
        async with AsyncWebCrawler() as crawler:
            result = await crawler.arun(url=url, config=config)
            if result != None and len(result.markdown) < 300:
                result = None
            return result
    except:
        return None
    
import nest_asyncio

nest_asyncio.apply()

async def scrape_multiple(url_list):
    results = []
    for url in url_list:
        result = await scrape_url(url)
        results.append(result)
    return results

def clean_multiple(results):
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-8b")
    st_llm = llm.with_structured_output(ArticleSchema)
    clean_md_prompt = """
    You are an advanced document processor. The input is a Markdown news article that contains unnecessary content, advertisements, and web elements.

    Your goal is to extract **only** the **main body** of the article while preserving key formatting.

    **Instructions:**
    - Keep only the **core article content** (remove headers, footers, sidebars, ads, related news, hyperlinks, comments, and irrelevant navigation).
    - Maintain the structure with **paragraphs, bullet points, and bold/italic formatting** if present.
    - DO NOT edit or summarize any of the content from the article. Ensure full and complete extraction of the main body of the article.
    """
    # Process each document
    cleaned_results = []
    for doc in results:
        try:
            prompt = f"{clean_md_prompt}\n\n{doc[0]}"
            result = st_llm.invoke(prompt)
            cleaned_results.append(result.main_content)
        except Exception as e:
            cleaned_results.append("")
    for n, c in enumerate(cleaned_results):
        try:
            if c != "" and len(c) <300:
                cleaned_results[n] = ""
            else:
                pass
        except:
            cleaned_results[n]= ""
    return cleaned_results


def chunk_document_for_sentiment(md_txt):
    n = len(md_txt)
    chunk_len = n//3
    return md_txt[chunk_len:2*chunk_len]

def compute_sentiment_score(spacy_model, title, md_txt):
    txt = remove_non_english(md_txt)
    txt = chunk_document_for_sentiment(txt)
    
    doc = spacy_model(title)
    headline_sentiment = doc._.blob.polarity

    doc = spacy_model(txt)
    chunk_polarity = doc._.blob.polarity
    score = headline_sentiment * 0.3 + chunk_polarity*0.7
    if score == 0:
        score = random.uniform(-0.2, 0.2)
    return score

def compute_multiple_sentiment_score(spacy_model, df, cleaned_results):
    sent_scores = []
    headlines = df['title'].tolist()
    # check later
    for n, h in enumerate(headlines):
        md_txt = cleaned_results[n]
        try:
            score = compute_sentiment_score(spacy_model, h, md_txt)
        except:
            score = random.uniform(-0.2, 0.2)
        sent_scores.append(score)
    return sent_scores

def extract_countries(cleaned_results):
    all_countries = []
    for c in cleaned_results:
        curr_countries = []
        for acs in abv_countries:
            for ac in acs:
                if ac in c:
                    curr_countries.append(acs[0])
                    break
                else:
                    pass
        for country in countries:
            if country in c:
                curr_countries.append(country)
        all_countries.append(("|").join(curr_countries))
    return all_countries

def st_visualize_sent(df):
    dates = df["seendate"].tolist()
    dates = [d[:8] for d in dates]
    df["seendate"] = dates
    
    df = df.sort_values(by="seendate")

    # Create the scatter plot
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(df["seendate"], df["sentiment_scores"], c="blue", edgecolors="black", alpha=0.75)

    # Formatting the plot
    ax.axhline(0, color="gray", linestyle="--")  # Add a horizontal line at sentiment = 0
    ax.set_xlabel("Datetime")
    ax.set_ylabel("Sentiment Score")
    ax.set_title("Sentiment Score Over Time (Color Coded)")
    ax.tick_params(axis='x', rotation=45)  # Rotate x-axis labels for readability
    ax.set_ylim(-1, 1)  # Set y-axis limits between -1 and 1
    ax.grid(True)

    # Display the plot in Streamlit
    st.pyplot(fig)


def visualize_country_mentions(df):
    # Extract and count occurrences of each country
    country_mentions = df["countries_mentioned"].dropna().str.split("|").explode()  # Split & flatten
    country_counts = Counter(country_mentions)  # Count occurrences

    # Convert to DataFrame for plotting
    country_df = pd.DataFrame(country_counts.items(), columns=["Country", "Article Count"])
    country_df = country_df.sort_values(by="Article Count", ascending=False)  # Sort in descending order

    # Plot the top countries
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(country_df["Country"][:10], country_df["Article Count"][:10], color="royalblue")  # Top 15 countries

    # Formatting the plot
    ax.set_xlabel("Country")
    ax.set_ylabel("Number of Articles Mentioning Country")
    ax.set_title("Top Countries Mentioned in Articles")
    ax.tick_params(axis='x', rotation=45)  # Rotate x-axis labels for readability

    # Display the plot in Streamlit
    st.pyplot(fig)



def load_md_as_doc(md_text, title, seendate, domain, url):
    """
    Converts Markdown text into LangChain Document objects with metadata.
    Splits text into chunks while tracking character positions.
    """

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)

    # Split document while tracking positions
    split_docs = []
    char_start = 0  # Track character position
    count = 1
    for split in text_splitter.split_text(md_text):
        char_end = char_start + len(split)  # Track split end position
        split_docs.append(
            Document(
                metadata={
                    "title": title,
                    "seendate": seendate,
                    "domain": domain,
                    "url": url,
                    'split_start': char_start,
                    'split_end': char_end,
                    'chunk_number': count,
                    'chunk_total': len(text_splitter.split_text(md_text))
                },
                page_content=split
            )
        )
        char_start += len(split) - 150
        count += 1
    return split_docs

def load_all_md_as_docs(df):
    """
    Converts a DataFrame containing cleaned Markdown text into LangChain Document objects.
    """
    all_docs = []
    for idx, row in df.iterrows():
        md_text = row['cleaned_txt']  # Assuming cleaned Markdown is in this column
        title = row['title']
        seendate = row['seendate']
        seendate = seendate[:8]
        domain = row['domain']
        url = row["url"]

        res = load_md_as_doc(md_text, title, seendate, domain, url)
        all_docs.extend(res)  # Append all split documents
    return all_docs


            

# Streamlit UI
st.title("MVP: VGEO")
st.subheader("🔍 Data Collection")
code = '''Sample Search Queries

trump AND (trade OR tariff) AND (congress OR senate OR "house of representatives" OR "political standoff") AND ("international relations" OR "diplomatic tensions")

trump AND (tariff OR trade) AND ("international relations" OR "diplomatic tensions") AND (Colombia OR Canada OR China OR EU OR Japan OR Mexico OR Korea)

("German election" OR "Bundestag election") AND ("Olaf Scholz" OR "Friedrich Merz" OR "Robert Habeck" OR "Annalena Baerbock" OR "Christian Lindner") AND (CDU OR CSU OR SPD OR Greens OR AfD OR FDP OR BSW)

Ukraine AND Russia AND (Putin OR Zelenskyy) AND (trump OR US OR USA OR EU OR NATO) AND ("peace talks" OR negotiation)

'''
st.code(code)

st.info("🔹 Enter a search query and select a valid date range to start. Recommended date range 1mo - 3mo.")
gdelt_search_query = None
start_date = None
end_date = None

# User input
gdelt_search_query = st.text_area("Enter Article Search Query", "")
gdelt_search_query = gdelt_search_query.strip()

start_date = st.date_input("Start Date").strftime("%Y%m%d")
end_date = st.date_input("End Date").strftime("%Y%m%d")

def st_search_gdelt(q, s_date, e_date):
    formatted_query = q.replace(" ", "%20")
    final_query = f'{formatted_query}&sourcelang=eng&startdatetime={s_date}000000&enddatetime={e_date}000000&maxrecords=250'
    
    # Debugging: Print the final query
    print("Final Query:", final_query)
    df= None
    try:# Execute search
        df = search_gdelt_queries(final_query)
    except:
        st.warning("⚠ There was an error in the search query. Try adjusting your query or date range.")

    if df is not None:
        len_df = len(df)
        st.info(f"🔹 Search Results: {str(len_df).zfill(3)} Articles were located. Adjust query or date range if you want to collect more.")
    return df 


df = None
if gdelt_search_query and start_date and end_date:
    df = st_search_gdelt(gdelt_search_query, start_date, end_date)


# **New Feature: Proceed with Scraping?**
proceed_with_scraping = False
if df is not None:
    proceed_with_scraping = st.radio(
        "Would you like to scrape the located articles?",
        ("No", "Yes"),
        index=0,  # Default set to "No"
        key = "scrape_continue"
    )


cleaned_results = None
if proceed_with_scraping == "Yes" and st.session_state.scraped_once == False:
    if df is not None and len(df) > 0:
        st.session_state.scraped_once = True
        st.info("🔄 Scraping articles... This may take some time.")

        # Get list of URLs
        url_list = df['url'].tolist()

        # Run async scraping
        scrape_results = asyncio.run(scrape_multiple(url_list))

        # loop = asyncio.new_event_loop()
        # asyncio.set_event_loop(loop)
        # scrape_results = loop.run_until_complete(scrape_multiple(url_list))
        print("cleaning...")
        cleaned_results = clean_multiple(scrape_results)
        print("done cleaning...")
    else:
        st.warning("⚠ No valid articles to scrape.")


nlp = spacy.load("en_core_web_sm")
nlp.add_pipe("spacytextblob")
scores = None
if cleaned_results is not None and len(df) == len(cleaned_results):
    len_og_df = len(df)
    df['platform'] = "gdelt"
    df['query'] = gdelt_search_query
    df['cleaned_txt'] = cleaned_results
    df["cleaned_txt"].replace("", np.nan, inplace=True)
    df = df.dropna(subset=["cleaned_txt"]).reset_index(drop=True)
    st.info(f"🔄 Scraped and Cleaned {len(df)} Articles Successfully.")
    scores = compute_multiple_sentiment_score(nlp, df, df['cleaned_txt'].tolist())
    countries = extract_countries(df['cleaned_txt'].tolist())
    df['sentiment_scores'] = scores
    df['countries_mentioned'] = countries
    st.dataframe(df)

visualized = False
if scores is not None:
    st_visualize_sent(df)
    visualize_country_mentions(df)
    visualized = True

vector_db = None
if visualized == True:
    all_docs = load_all_md_as_docs(df)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    # Connect to Milvus/Zilliz
    connection_args = {
        "uri": os.getenv("ZILLIZ_CLOUD_URI"),
        "user": os.getenv("ZILLIZ_CLOUD_USERNAME"),
        "password": os.getenv("ZILLIZ_CLOUD_PASSWORD"),
        "secure": True,
    }
    
    client = MilvusClient(
        uri=os.getenv("ZILLIZ_CLOUD_URI"),
        token=os.getenv("ZILLIZ_CLOUD_API_KEY")
    )
    client.drop_collection(
        collection_name="LangChainCollection"
    )
    # Store in Milvus/Zilliz
    vector_db = Milvus.from_documents(all_docs, embeddings, collection_name= f"LangChainCollection",connection_args=connection_args)

if vector_db is not None:
    vector_search_query = st.text_area("Enter Vector Search Query", "")
    vector_search_query = vector_search_query.strip()
    vector_df = None
    if vector_search_query != "":
        docs = vector_db.similarity_search(vector_search_query)
        all_rows=[]
        for doc in docs:
            row = [doc.metadata.get("title"), doc.metadata.get("seendate")[:8], doc.metadata.get("url"), doc.page_content[:30]]
            all_rows.append(row)
        vector_df = pd.DataFrame(data = all_rows, columns = ["title", "seendate", "url", "preview"])

if visualized == True and vector_db is not None:
    final_df = df[['title', 'seendate', 'domain', 'url', 'language', 'sourcecountry', 'platform', 'query', 'sentiment_scores', 'countries_mentioned', 'cleaned_txt']]
    
    st.subheader("📥 Download Processed Data as Excel")

    # Create an in-memory Excel file
    excel_buffer = io.BytesIO()
    with pd.ExcelWriter(excel_buffer, engine="xlsxwriter") as writer:
        final_df.to_excel(writer, index=False)

    # Provide download button in Streamlit
    st.download_button(
        label="Download Excel File",
        data=excel_buffer.getvalue(),
        file_name="articles_data.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
# Footer
st.markdown("---")


