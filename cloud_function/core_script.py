def run_pipeline():


    from sqlalchemy import create_engine, text
    import pandas as pd
    import requests
    from datetime import date, timedelta, datetime
    import openai
    import json
    import time
    import os
    import logging
    from dotenv import load_dotenv

    logging.basicConfig(level=logging.INFO)
    logging.info("🔁 Starting Wikipedia pipeline job")

    # --- Setup ---
    load_dotenv()
    openai.api_key = os.getenv("OPENAI_API_KEY")
    if not openai.api_key:
        raise ValueError("OPENAI_API_KEY not set in environment variables.")

    CATEGORIES = [
        "Entertainment", "Movies & TV", "Music", "Sports", "Politics", "Business",
        "Crime & Justice", "Technology", "Science", "Health", "Education",
        "History", "Culture", "Religion", "Military",
        "Visual Arts & Photography", "Travel", "Food & Beverage", "Fashion",
        "Literature", "Video Games", "Geography", "General Knowledge"
    ]

    blacklist_titles = ["Main_Page", "विशेष:अलीकडील_बदल"]
    blacklist_prefixes = ["Special:", "Wikipedia:", "Portal:", "Wiktionary:", "Help:", "File:", "Wikipédia:"]

    def get_engine():
        db_url = (
            f"postgresql://{os.getenv('PG_USER')}:{os.getenv('PG_PASSWORD')}"
            f"@{os.getenv('PG_HOST')}:{os.getenv('PG_PORT')}/{os.getenv('PG_DB')}"
        )
        return create_engine(db_url, connect_args={"sslmode": "require"})

    def get_cached_metadata_pg(conn, title):
        result = conn.execute(
            text("SELECT description, wiki_categories FROM article_metadata WHERE title = :title"),
            {"title": title}
        ).fetchone()
        if result:
            description, wiki_categories = result
            return description, json.loads(wiki_categories)
        return None, None

    def cache_metadata_pg(conn, title, description, wiki_categories):
        conn.execute(
            text("""
                INSERT INTO article_metadata (title, description, wiki_categories)
                VALUES (:title, :description, :wiki_categories)
                ON CONFLICT (title)
                DO UPDATE SET description = EXCLUDED.description,
                            wiki_categories = EXCLUDED.wiki_categories
            """),
            {
                "title": title,
                "description": description,
                "wiki_categories": json.dumps(wiki_categories)
            }
        )

    def get_cached_categories_pg(conn, title):
        result = conn.execute(
            text("SELECT categories FROM gpt_classifications WHERE title = :title"),
            {"title": title}
        ).fetchone()
        if result:
            return json.loads(result[0])
        return None

    def cache_categories_pg(conn, title, categories):
        conn.execute(
            text("""
                INSERT INTO gpt_classifications (title, categories)
                VALUES (:title, :categories)
                ON CONFLICT (title)
                DO UPDATE SET categories = EXCLUDED.categories
            """),
            {
                "title": title,
                "categories": json.dumps(categories)
            }
        )

    def save_to_postgres(conn, df):
        df["gpt_categories"] = df["gpt_categories"].apply(json.dumps)
        df["wiki_categories"] = df["wiki_categories"].apply(json.dumps)
        df.to_sql("articles", conn.engine, if_exists="append", index=False, method="multi")

    def get_short_description(title):
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{title}"
        try:
            response = requests.get(url)
            if response.status_code == 200:
                return response.json().get("description", "")
        except:
            pass
        return ""

    def get_visible_categories(title):
        url = "https://en.wikipedia.org/w/api.php"
        params = {
            "action": "query",
            "format": "json",
            "prop": "categories",
            "titles": title,
            "cllimit": "max",
            "clshow": "!hidden"
        }
        try:
            response = requests.get(url, params=params)
            data = response.json()
            pages = data.get("query", {}).get("pages", {})
            page = next(iter(pages.values()))
            raw_cats = page.get("categories", [])
            return [cat['title'].replace("Category:", "") for cat in raw_cats]
        except:
            return []

    def classify_rows_multicategory(df, chunk_size=20, sleep_time=0.5):
        results = []
        chunks = [df.iloc[i:i + chunk_size] for i in range(0, len(df), chunk_size)]

        for chunk in chunks:
            items = []
            for _, row in chunk.iterrows():
                title = row["article"]
                description = str(row.get("description", "")).strip()
                items.append({"title": title, "description": description} if description else {"title": title})

            if not items:
                continue

            prompt = (
                "You are a content strategist classifying Wikipedia articles. "
                "You must assign **only** 1 to 3 categories to each article from the following fixed list:\n\n"
                + ", ".join(CATEGORIES) + "\n\n"
                "Respond in this exact JSON format:\n"
                "[{\"title\": \"Article Title\", \"categories\": [\"Category1\", \"Category2\"]}, ...]\n\n"
                "Here are the articles to classify:\n"
                + json.dumps(items, indent=2)
            )

            try:
                response = openai.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0
                )
                parsed = json.loads(response.choices[0].message.content)

                for entry in parsed:
                    title = entry["title"]
                    raw_categories = entry.get("categories", [])
                    valid_categories = [cat for cat in raw_categories if cat in CATEGORIES]
                    if not valid_categories:
                        valid_categories = ["Uncategorized"]
                    results.append({"title": title, "categories": valid_categories})

            except Exception as e:
                for item in items:
                    logging.warning(f"⚠️ GPT classification failed: {e}")
                    results.append({"title": item["title"], "categories": ["Uncategorized"]})

            time.sleep(sleep_time)

        return pd.DataFrame(results)

    def clean_title(title):
        return title.strip().replace("_", " ")

    yesterday = date.today() - timedelta(days=1)
    yesterday_str = yesterday.strftime('%Y/%m/%d')

    headers = {'User-Agent': 'Mozilla/5.0'}
    url = f'https://wikimedia.org/api/rest_v1/metrics/pageviews/top-per-country/US/mobile-web/{yesterday_str}'
    res = requests.get(url, headers=headers)
    articles = res.json()['items'][0]['articles']
    df = pd.DataFrame(articles)

    df_filtered = df[
        ~df['article'].isin(blacklist_titles) &
        ~df['article'].str.startswith(tuple(blacklist_prefixes))
    ][['article', 'views_ceil', 'rank']].rename(columns={'views_ceil': 'views'})

    df_filtered['link'] = 'https://en.wikipedia.org/wiki/' + df_filtered['article']
    df_final = df_filtered[['rank', 'article', 'views', 'link']].copy()
    df_final["article"] = df_final["article"].apply(clean_title)

    descriptions, wiki_cats, gpt_cats, to_classify = [], [], [], []
    engine = get_engine()

    with engine.begin() as conn:
        for _, row in df_final.iterrows():
            title = row['article']
            slug = row['link'].split('/wiki/')[-1]

            desc, wiki_cat = get_cached_metadata_pg(conn, title)
            if desc is None:
                desc = get_short_description(slug)
                wiki_cat = get_visible_categories(slug)
                cache_metadata_pg(conn, title, desc, wiki_cat)
            descriptions.append(desc)
            wiki_cats.append(wiki_cat)

            cached_cats = get_cached_categories_pg(conn, title)
            if cached_cats is None:
                to_classify.append({"title": title, "description": desc})
                gpt_cats.append(None)
            else:
                gpt_cats.append(cached_cats)

        df_final["description"] = descriptions
        df_final["wiki_categories"] = wiki_cats
        df_final["gpt_categories"] = gpt_cats

        if to_classify:
            df_to_classify = pd.DataFrame(to_classify)
            df_to_classify.rename(columns={"title": "article"}, inplace=True)
            gpt_df = classify_rows_multicategory(df_to_classify)

            gpt_df.dropna(subset=["title", "categories"], inplace=True)
            gpt_df = gpt_df[gpt_df["categories"].map(lambda x: isinstance(x, list))]

            for _, row in gpt_df.iterrows():
                cache_categories_pg(conn, row["title"], row["categories"])

            categories_df = gpt_df.rename(columns={"title": "article", "categories": "gpt_categories"})
            df_final = df_final.merge(categories_df, on="article", how="left")

            df_final["gpt_categories"] = df_final.apply(
                lambda row: row["gpt_categories_y"] if pd.isna(row["gpt_categories_x"]) else row["gpt_categories_x"],
                axis=1
            )
            df_final = df_final.drop(columns=["gpt_categories_x", "gpt_categories_y"])

        df_final = df_final.sort_values("views", ascending=False).reset_index(drop=True)
        df_final["date"] = datetime.strptime(yesterday_str, "%Y/%m/%d").date().isoformat()
        df_final["slug"] = df_final["link"].apply(lambda x: x.split("/wiki/")[-1])
        df_final["id"] = df_final["date"] + "::" + df_final["slug"]

        df_final = df_final[[
            "date", "article", "views", "link", "description",
            "gpt_categories", "wiki_categories", "slug", "id"
        ]]

        save_to_postgres(conn, df_final)
        logging.info(f"✅ Successfully processed {len(df_final)} articles on {df_final['date'].iloc[0]}")