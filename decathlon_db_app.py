import streamlit as st
import pandas as pd
import sqlite3
from transformers import pipeline
from unidecode import unidecode

# --- DATABÁZOVÁ ARCHITEKTÚRA ---
def init_advanced_db():
    conn = sqlite3.connect('decathlon_warehouse.db')
    c = conn.cursor()
    
    # 1. BRONZOVÁ: Surové dáta zo scrapingu
    c.execute('''CREATE TABLE IF NOT EXISTS raw_reviews 
                 (review_id TEXT PRIMARY KEY, pobocka TEXT, author TEXT, text TEXT, stars INTEGER, date TEXT)''')
    
    # 2. STRIEBORNÁ: Celkový sentiment recenzie
    c.execute('''CREATE TABLE IF NOT EXISTS processed_sentiment 
                 (review_id TEXT PRIMARY KEY, sentiment_label TEXT, sentiment_score REAL,
                  FOREIGN KEY(review_id) REFERENCES raw_reviews(review_id))''')
    
    # 3. ZLATÁ: Rozsekané aspekty (jedna recenzia môže mať viac riadkov)
    c.execute('''CREATE TABLE IF NOT EXISTS aspect_analysis 
                 (aspect_id INTEGER PRIMARY KEY AUTOINCREMENT, review_id TEXT, 
                  aspekt TEXT, veta TEXT, sentiment TEXT,
                  FOREIGN KEY(review_id) REFERENCES raw_reviews(review_id))''')
    
    conn.commit()
    conn.close()

# --- SIMULÁCIA SCRAPINGU ---
def simulate_scraping():
    # Simulujeme, čo by robot "vyškriabal" z webu
    scraped_data = [
        ("ID_001", "Bratislava", "Peter M.", "Bicykel je super, ale obsluha bola pomalá.", 4, "2024-05-20"),
        ("ID_002", "Žilina", "Anna K.", "Hrozný neporiadok na predajni a drahý tovar.", 2, "2024-05-21"),
        ("ID_003", "Košice", "Marek", "Milý personal, skvelé ceny.", 5, "2024-05-22")
    ]
    conn = sqlite3.connect('decathlon_warehouse.db')
    for r in scraped_data:
        try:
            conn.execute("INSERT INTO raw_reviews VALUES (?,?,?,?,?,?)", r)
        except: pass
    conn.commit()
    conn.close()

# --- HLAVNÁ ANALÝZA (Z Bronzu do Zlata) ---
@st.cache_resource
def get_nlp():
    return pipeline("sentiment-analysis", model="cardiffnlp/twitter-xlm-roberta-base-sentiment")

def process_data_warehouse():
    nlp = get_nlp()
    conn = sqlite3.connect('decathlon_warehouse.db')
    
    # Nájdeme recenzie, ktoré ešte nie sú v Striebornej tabuľke
    raw_to_process = pd.read_sql('''SELECT * FROM raw_reviews WHERE review_id NOT IN 
                                    (SELECT review_id FROM processed_sentiment)''', conn)
    
    label_map = {'positive': 'Pozitívny', 'neutral': 'Neutrálny', 'negative': 'Negatívny'}
    
    for _, row in raw_to_process.iterrows():
        # 1. Zapíšeme celkový sentiment (STRIEBRO)
        res = nlp(row['text'])[0]
        conn.execute("INSERT INTO processed_sentiment VALUES (?,?,?)", 
                     (row['review_id'], label_map[res['label']], res['score']))
        
        # 2. Aspektová analýza (ZLATO) - jednoduché rozdelenie
        vety = row['text'].split(',') # Simulácia rozdelenia na klauzuly
        for veta in vety:
            clean_veta = unidecode(veta.lower())
            found_asp = None
            if "obsluha" in clean_veta or "personal" in clean_veta: found_asp = "Personál"
            if "bicykel" in clean_veta or "tovar" in clean_veta: found_asp = "Produkty"
            if "cena" in clean_veta or "drahy" in clean_veta: found_asp = "Cena"
            
            if found_asp:
                veta_res = nlp(veta)[0]
                conn.execute("INSERT INTO aspect_analysis (review_id, aspekt, veta, sentiment) VALUES (?,?,?,?)",
                             (row['review_id'], found_asp, veta.strip(), label_map[veta_res['label']]))
    
    conn.commit()
    conn.close()

# --- UI ---
st.title("🏗️ Decathlon Data Warehouse")
init_advanced_db()

col1, col2 = st.columns(2)
with col1:
    if st.button("1. Spustiť Scraper (Bronze)"):
        simulate_scraping()
        st.success("Surové recenzie stiahnuté.")

with col2:
    if st.button("2. Spracovať dáta (Silver & Gold)"):
        process_data_warehouse()
        st.success("Dátový sklad bol aktualizovaný.")

# Zobrazenie Zlatej vrstvy (to, čo chce manažér)
st.subheader("🏆 Zlatá vrstva: Finálna analýza pre Dashboard")
conn = sqlite3.connect('decathlon_warehouse.db')
gold_df = pd.read_sql('''
    SELECT r.pobocka, a.aspekt, a.sentiment, a.veta 
    FROM aspect_analysis a
    JOIN raw_reviews r ON a.review_id = r.review_id
''', conn)
conn.close()

st.table(gold_df)