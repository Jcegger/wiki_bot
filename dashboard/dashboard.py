import streamlit as st
import pandas as pd
import json
from datetime import timedelta, date
import requests
import math
import os
from sqlalchemy import create_engine
import streamlit.components.v1 as components

def get_secret(key):
    return st.secrets.get(key) or os.getenv(key)

PG_USER = get_secret("PG_USER")
PG_PASSWORD = get_secret("PG_PASSWORD")
PG_HOST = get_secret("PG_HOST")
PG_PORT = get_secret("PG_PORT")
PG_DB = get_secret("PG_DB")
SERPAPI_KEY = get_secret("SERPAPI_KEY")

# --- Config ---
PAGE_SIZE = 25

# --- Database Utilities ---
def get_engine():
    db_url = (
        f"postgresql://{PG_USER}:{PG_PASSWORD}"
        f"@{PG_HOST}:{PG_PORT}/{PG_DB}"
    )
    return create_engine(db_url, connect_args={"sslmode": "require"})

@st.cache_data(show_spinner=False)
def get_available_dates():
    engine = get_engine()
    df = pd.read_sql("SELECT DISTINCT date FROM articles ORDER BY date", engine)
    df["date"] = pd.to_datetime(df["date"])
    return df["date"].tolist()

@st.cache_data(show_spinner=False)
def load_data(start_date, end_date):
    engine = get_engine()
    query = """
        SELECT id, date, slug, article, views, link, description,
               gpt_categories, wiki_categories
        FROM articles
        WHERE date BETWEEN %(start)s AND %(end)s
    """

    df = pd.read_sql(query, engine, params={"start": str(start_date), "end": str(end_date)})
    df["gpt_categories"] = df["gpt_categories"].apply(json.loads)
    df["wiki_categories"] = df["wiki_categories"].apply(json.loads)
    df["date"] = pd.to_datetime(df["date"])
    return df

@st.cache_data(show_spinner=False)
def get_days_ranked_map():
    engine = get_engine()
    query = """
        SELECT article, COUNT(DISTINCT date) AS days_ranked
        FROM articles
        GROUP BY article
    """
    df = pd.read_sql(query, engine)
    return dict(zip(df["article"], df["days_ranked"]))

# --- Image Fetcher ---
@st.cache_data(show_spinner=False)
def get_wikipedia_image(title):
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{title}"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            return data.get("thumbnail", {}).get("source")
    except:
        return None

# --- Full Summary Fetcher ---
@st.cache_data(show_spinner=False)
def get_full_summary(slug):
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{slug}"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            return data.get("extract")
    except:
        return "Unable to fetch summary."

# --- Top Stories Fetcher ---    
@st.cache_data(show_spinner=False)
def get_top_stories(query, num_results=3):
    try:
        response = requests.get(
            "https://serpapi.com/search",
            params={
                "q": query,
                "tbm": "nws",
                "api_key": SERPAPI_KEY,
                "num": num_results
            }
        )
        if response.status_code == 200:
            results = response.json().get("news_results", [])[:num_results]
            return [
                {
                    "title": item.get("title"),
                    "link": item.get("link"),
                    "source": item.get("source"),
                    "date": item.get("date")
                }
                for item in results
            ]
    except:
        return []    

# --- Related Questions Fetcher ---    
@st.cache_data(show_spinner=False)
def get_related_questions(query):
    try:
        response = requests.get(
            "https://serpapi.com/search",
            params={
                "q": query,
                "api_key": SERPAPI_KEY
            }
        )
        if response.status_code == 200:
            return response.json().get("related_questions", [])
    except:
        return []

# --- Sidebar Filters ---
st.sidebar.title("\U0001F4C5 Filters")
available_dates = get_available_dates()
min_date, max_date = available_dates[0], available_dates[-1]

date_filter = st.sidebar.selectbox(
    "Select Date Range",
    ["Yesterday", "Last 7 Days", "Last 28 Days", "Custom Range"],
    index=0
)

if date_filter == "Yesterday":
    start_date = end_date = max_date.date()
elif date_filter == "Last 7 Days":
    start_date = (max_date - timedelta(days=6)).date()
    end_date = max_date.date()
elif date_filter == "Last 28 Days":
    start_date = (max_date - timedelta(days=27)).date()
    end_date = max_date.date()
elif date_filter == "Custom Range":
    start_date = st.sidebar.date_input("Start Date", value=max_date - timedelta(days=6), min_value=min_date, max_value=max_date)
    end_date = st.sidebar.date_input("End Date", value=max_date, min_value=min_date, max_value=max_date)
    if start_date > end_date:
        st.sidebar.error("Start date must be before end date.")
        st.stop()

# --- Load filtered data ---
df = load_data(start_date, end_date)
df_filtered = df.copy()

# Always aggregate data
df_filtered = (
    df_filtered.sort_values("date", ascending=False)
    .groupby("article", as_index=False)
    .agg({
        "views": "sum",
        "link": "first",
        "description": "first",
        "gpt_categories": "first",
        "wiki_categories": "first",
        "slug": "first"
    })
)

# --- Days Ranked Calculation ---
days_ranked_map = get_days_ranked_map()
df_filtered["days_ranked"] = df_filtered["article"].map(days_ranked_map).fillna(1).astype(int)

# --- Rank Calculation ---
df_filtered = df_filtered.sort_values("views", ascending=False).reset_index(drop=True)
df_filtered["rank"] = df_filtered.index + 1

# --- Filters ---
all_categories = sorted({cat for sublist in df["gpt_categories"] for cat in sublist})
selected_category = st.sidebar.selectbox("Filter by Category", ["All"] + all_categories)

if selected_category != "All":
    df_filtered = df_filtered[df_filtered["gpt_categories"].apply(lambda cats: selected_category in cats)]

search_query = st.sidebar.text_input("Search by Article Title")
if search_query:
    df_filtered = df_filtered[df_filtered["article"].str.contains(search_query, case=False)]

# --- Reset Pagination if Filter Changed ---
if "last_filter_state" not in st.session_state:
    st.session_state.last_filter_state = {
        "date_filter": date_filter,
        "start_date": start_date,
        "end_date": end_date,
        "category": selected_category,
        "search": search_query
    }

if (
    st.session_state.last_filter_state["date_filter"] != date_filter or
    st.session_state.last_filter_state["start_date"] != start_date or
    st.session_state.last_filter_state["end_date"] != end_date or
    st.session_state.last_filter_state["category"] != selected_category or
    st.session_state.last_filter_state["search"] != search_query
):
    st.session_state.current_page = 1
    st.session_state.last_filter_state = {
        "date_filter": date_filter,
        "start_date": start_date,
        "end_date": end_date,
        "category": selected_category,
        "search": search_query
    }

# --- Pagination Setup ---
total_articles = len(df_filtered)
total_pages = max(1, math.ceil(total_articles / PAGE_SIZE))
if "current_page" not in st.session_state:
    st.session_state.current_page = 1

st.session_state.current_page = min(
    max(1, st.session_state.get("current_page", 1)),
    total_pages
)

# --- View Mode Toggle ---
view_mode = st.radio(
    "Select View Mode",
    ["Detailed View", "Table View"],
    index=0,
    horizontal=True
)

# --- Display Title & Caption ---
st.title("\U0001F4F0 Top Wikipedia Articles (US Mobile Web)")
st.caption(f"Showing articles {(st.session_state.current_page - 1) * PAGE_SIZE + 1}–{min(st.session_state.current_page * PAGE_SIZE, total_articles)} of {total_articles} from **{start_date} to {end_date}**")

# --- Pagination Slice ---
start_idx = (st.session_state.current_page - 1) * PAGE_SIZE
end_idx = start_idx + PAGE_SIZE
page_df = df_filtered.iloc[start_idx:end_idx]

# --- Image Banner for Top Article ---
if not df_filtered.empty:
    top_article_row = df_filtered.sort_values("views", ascending=False).iloc[0]
    image_url = get_wikipedia_image(top_article_row["slug"])
    if image_url:
        st.markdown(
            f"<div style='text-align: center; margin-bottom: 1rem;'>"
            f"<img src='{image_url}' style='max-width:100%'; height=auto;'/>"
            f"</div>",
            unsafe_allow_html=True
        )

# --- CSV Download Button ---
download_df = df_filtered.copy()
download_df["views"] = download_df["views"].apply(lambda x: f"{x:,}")
download_df["article_link"] = download_df.apply(
    lambda row: f'https://en.wikipedia.org/wiki/{row["slug"]}', axis=1
)

csv = download_df[["rank", "article", "views", "description", "days_ranked", "article_link"]].to_csv(index=False)
st.download_button(
    label="\U0001F4C5 Download Full Table as CSV",
    data=csv,
    file_name=f"wikipedia_articles_{start_date}_to_{end_date}.csv",
    mime="text/csv"
)

if view_mode == "Detailed View":
    for idx, row in page_df.iterrows():
        st.markdown(f"### {row['rank']}. [{row['article']}]({row['link']})")
        st.write(f"**Views**: {row['views']:,} | **Days Ranked**: {row['days_ranked']}")
        st.write(row['description'])

        with st.expander("📖 Full Description"):
            full_summary = get_full_summary(row["slug"])
            st.write(full_summary)

        # --- Toggle Logic ---
        insight_key = f"insights_visible_{row['slug']}"
        button_key = f"toggle_button_{row['slug']}"

        if insight_key not in st.session_state:
            st.session_state[insight_key] = False

        # Toggle button
        if st.button(
            "📊 Show Article Insights" if not st.session_state[insight_key] else "🙈 Hide Article Insights",
            key=button_key
        ):
            st.session_state[insight_key] = not st.session_state[insight_key]
            st.experimental_rerun()  # Ensures the button label immediately reflects the new state

        # Render content if toggled on
        if st.session_state[insight_key]:
            with st.spinner("Fetching insights..."):

                # --- Top Stories ---
                st.subheader("📰 Top Stories")
                top_stories = get_top_stories(row["article"])
                if not top_stories:
                    st.info("No news stories found.")
                else:
                    for story in top_stories:
                        st.markdown(
                            f"- [{story['title']}]({story['link']}) — *{story['source']}, {story['date']}*"
                        )

                # --- Related Questions ---
                st.subheader("❓ Related Questions")
                related = get_related_questions(row["article"])
                if not related:
                    st.info("No related questions found.")
                else:
                    for q in related:
                        st.markdown(f"- {q.get('question')}")

        st.markdown("---")

else:
    table_df = page_df[["rank", "article", "link", "views", "days_ranked", "description"]].copy()
    table_df["views"] = table_df["views"].apply(lambda x: f"{x:,}")
    table_df["article"] = table_df.apply(
        lambda row: f'<a href="{row["link"]}" target="_blank">{row["article"]}</a>',
        axis=1
    )

    # Drop 'link' now that it's embedded in 'article'
    table_df = table_df.drop(columns=["link"])
    
    table_df = table_df.rename(columns={
        "rank": "Rank",
        "article": "Article",
        "views": "Views",
        "days_ranked": "Days Ranked",
        "description": "Description"
    })
    st.write(table_df.to_html(escape=False, index=False), unsafe_allow_html=True)

# --- Pagination Controls ---
col1, col2, col3 = st.columns([1, 2, 1])

with col1:
    if st.button("⬅️ Previous", key="prev_page"):
        st.session_state.current_page = max(1, st.session_state.current_page - 1)
        st.experimental_rerun()

with col3:
    if st.button("Next ➡️", key="next_page"):
        st.session_state.current_page = min(total_pages, st.session_state.current_page + 1)
        st.experimental_rerun()

with col2:
    page_options = list(range(1, total_pages + 1))
    current_index = (
        page_options.index(st.session_state.current_page)
        if st.session_state.current_page in page_options
        else 0
    )
    selected_page = st.selectbox(
        f"Page {st.session_state.current_page} of {total_pages}",
        options=page_options,
        index=current_index,
        key="page_selector_value",
        format_func=lambda x: f"Page {x}",
    )
    if selected_page != st.session_state.current_page:
        st.session_state.current_page = selected_page
        st.experimental_rerun()

# --- Top Article per Category ---
st.markdown("## \U0001F3C5 Top Article in Each Category")

top_by_cat = df_filtered.explode("gpt_categories").copy()
top_by_cat = (
    top_by_cat
    .sort_values("views", ascending=False)
    .drop_duplicates(subset=["gpt_categories"])
)

top_by_cat["Views"] = top_by_cat["views"].apply(lambda x: f"{x:,}")
top_by_cat["article_link"] = top_by_cat.apply(
    lambda row: f'<a href="{row["link"]}" target="_blank">{row["article"]}</a>', axis=1
)

top_cat_table = top_by_cat[["gpt_categories", "article_link", "Views", "description"]]
top_cat_table = top_cat_table.rename(columns={"gpt_categories": "Category"})

st.write(
    top_cat_table.to_html(escape=False, index=False),
    unsafe_allow_html=True
)