import streamlit as st
from transformers import pipeline
import re
import pandas as pd
import plotly.express as px
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import simplemma
from unidecode import unidecode
from thefuzz import fuzz
import sqlite3
from datetime import datetime

# --- 1. Nastavenie stránky ---
st.set_page_config(
    page_title="Decathlon Analytics",
    page_icon="🔬",
    layout="wide"
)

# --- 2. Načítanie modelu a NLTK dát ---
@st.cache_resource
def load_nlp_tools():
    model_path = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
    sentiment_model = pipeline("sentiment-analysis", model=model_path, tokenizer=model_path, top_k=None)
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        nltk.download('punkt')
        nltk.download('punkt_tab')
    return sentiment_model

# --- 3. Lematizácia ---
def lemmatize_text(text):
    words = word_tokenize(text, language='slovene')
    lemmatized_words = [simplemma.lemmatize(word, lang='sk') for word in words]
    return " ".join(lemmatized_words).lower()

# --- 4. Lexikón ---
ASPEKTOVE_SADY = {
    'Personál a Obsluha': ['personál', 'obsluha', 'predavač', 'predavačka', 'zamestnanec', 'poradca', 'pokladňa', 'prístup', 'ochota', 'poradiť', 'rad'],
    'Cena a Hodnota': ['cena', 'drahý', 'lacný', 'zľava', 'výpredaj', 'akcia', 'pomer', 'kvalita', 'peniaz', 'účet'],
    'Dostupnosť a E-shop': ['doprava', 'doručenie', 'kuriér', 'rýchlo', 'neskoro', 'balík', 'dodanie', 'eshop', 'objednávka', 'sklad', 'vyzdvihnutie', 'stránka', 'dostupnosť'],
    'Predajňa a Prostredie': ['prostredie', 'predajňa', 'obchod', 'pobočka', 'čistota', 'atmosféra', 'miesto', 'parkovanie', 'kabínka', 'výber', 'sortiment'],
    'Kvalita Produktov (Vybavenie a Oblečenie)': ['produkt', 'tovar', 'výrobok', 'materiál', 'oblečenie', 'obuv', 'topánka', 'vybavenie', 'pokazený', 'reklamácia', 'vydržať', 'rozpadnúť', 'strih', 'číslo', 'bicykel', 'stan', 'bunda', 'tričko', 'veľkosť']
}

# --- 5. Analytické funkcie ---
def process_sentiment_results(results):
    scores = {"Pozitívny": 0.0, "Neutrálny": 0.0, "Negatívny": 0.0}
    label_map = {"positive": "Pozitívny", "label_2": "Pozitívny", "neutral": "Neutrálny", "label_1": "Neutrálny", "negative": "Negatívny", "label_0": "Negatívny"}
    for res in results:
        label = res['label'].lower()
        slovak_label = label_map.get(label, label)
        if slovak_label in scores:
            scores[slovak_label] = res['score']
    max_label = max(scores, key=scores.get)
    return {"scores": scores, "label": max_label, "max_score": scores[max_label]}

def extract_aspects_ultimate(text, model):
    found_aspects = []
    sentences = sent_tokenize(text, language='slovene')
    CONJUNCTIONS = r'\b(ale|avšak|no|zatial čo|len|hoci|pritom|na druhej strane)\b'
    for sentence in sentences:
        clauses = re.split(CONJUNCTIONS, sentence, flags=re.IGNORECASE)
        processed_clauses = [clauses[0]]
        if len(clauses) > 1:
            for i in range(1, len(clauses), 2):
                processed_clauses.append(clauses[i] + clauses[i+1])
        for clause in processed_clauses:
            clause = clause.strip()
            if not clause: continue
            lemmatized_clause = lemmatize_text(clause)
            clause_normalized = unidecode(lemmatized_clause)
            clause_words = clause_normalized.split()
            for aspect_name, keywords in ASPEKTOVE_SADY.items():
                aspect_found = False
                for keyword in keywords:
                    keyword_norm = unidecode(keyword)
                    if keyword_norm in clause_normalized:
                        aspect_found = True
                        break
                    for word in clause_words:
                        if fuzz.ratio(keyword_norm, word) >= 80:
                            aspect_found = True
                            break
                    if aspect_found: break
                if aspect_found:
                    raw_result = model(clause)[0]
                    sentiment_data = process_sentiment_results(raw_result)
                    found_aspects.append({"Aspekt": aspect_name, "Časť vety": clause, "Zistený stav": sentiment_data['label']})
    return found_aspects

# --- 6. NOVÁ FUNKCIA: JEDNODUCHÉ TABUĽKY ---
def draw_summary_tables(df_overall, df_aspects):
    st.markdown("## 🏆 Manažérsky Súhrn (Executive Summary)")
    st.markdown("Prehľadné tabuľky s výsledkami.")

    # 1. KPI TABUĽKA (Celkový sentiment)
    st.subheader("1. Celková úspešnosť pobočiek")
    
    # Výpočet percent
    summary = df_overall.groupby('pobocka')['Sentiment'].value_counts(normalize=True).unstack().fillna(0) * 100
    summary = summary.rename(columns={'Pozitívny': 'Pozitívne %', 'Negatívny': 'Negatívne %', 'Neutrálny': 'Neutrálne %'})
    
    # Pridanie počtu recenzií
    counts = df_overall['pobocka'].value_counts()
    summary['Počet recenzií'] = counts
    
    # Zoradenie stĺpcov a riadkov
    cols = ['Počet recenzií', 'Pozitívne %', 'Neutrálne %', 'Negatívne %']
    existing_cols = [c for c in cols if c in summary.columns]
    summary = summary[existing_cols].sort_values(by='Pozitívne %', ascending=False)

    # Vykreslenie ČISTEJ tabuľky (len formátovanie čísel)
    st.dataframe(
        summary.style.format("{:.1f}%", subset=[c for c in existing_cols if '%' in c]),
        use_container_width=True
    )

    # 2. ASPEKTOVÁ MATICA
    if not df_aspects.empty:
        st.subheader("2. Detailná matica spokojnosti podľa aspektov")
        st.markdown("Tabuľka ukazuje **% pozitívnych recenzií** pre daný aspekt.")

        # Výpočet % spokojnosti pre každý aspekt
        aspect_totals = df_aspects.groupby(['pobocka', 'Aspekt']).size()
        aspect_positives = df_aspects[df_aspects['Sentiment'] == 'Pozitívny'].groupby(['pobocka', 'Aspekt']).size()
        
        aspect_matrix = (aspect_positives / aspect_totals * 100).unstack()
        aspect_matrix = aspect_matrix.fillna(0) # 0% ak nie sú pozitívne zmienky
        
        # Vykreslenie ČISTEJ tabuľky
        st.dataframe(
            aspect_matrix.style.format("{:.0f}%"),
            use_container_width=True
        )
    else:
        st.info("Pre zvolené kritériá neboli nájdené žiadne detailné aspekty.")

# --- 7. HLAVNÝ DASHBOARD ---
def main():
    st.title("📊 Decathlon Analytics: Analýza pobočiek")
    st.markdown("Vyberte si kritériá a systém zanalyzuje nové recenzie alebo načíta hotový report.")

    db_path = 'decathlon_warehouse.db'
    
    with sqlite3.connect(db_path) as conn:
        try:
            df_info = pd.read_sql_query("SELECT DISTINCT pobocka FROM raw_reviews WHERE pobocka IS NOT NULL AND pobocka != ''", conn)
            vsetky_pobocky = df_info['pobocka'].tolist()
            
            df_dates = pd.read_sql_query("SELECT date FROM raw_reviews WHERE date IS NOT NULL AND date != ''", conn)
            valid_dates = pd.to_datetime(df_dates['date'], errors='coerce').dropna()
            
            if not valid_dates.empty:
                db_min_date = valid_dates.min().date()
                db_max_date = valid_dates.max().date()
            else:
                db_min_date = db_max_date = datetime.today().date()
            
        except Exception as e:
            vsetky_pobocky = []
            db_min_date = db_max_date = datetime.today().date()
            st.warning(f"Chyba pri čítaní databázy pre filtre: {e}")

    # --- FILTRE PRIAMO NA STRÁNKE ---
    if not vsetky_pobocky:
        st.warning("⚠️ Databáza je prázdna, nie sú k dispozícii žiadne pobočky na filtrovanie.")
        return

    col_filter1, col_filter2 = st.columns(2)
    
    with col_filter1:
        vybrane_pobocky = st.multiselect("📍 Vyberte pobočky na analýzu:", options=vsetky_pobocky)
        
    with col_filter2:
        vybrany_datum = st.date_input("📅 Vyberte obdobie (Od - Do):", value=(db_min_date, db_max_date))
        analyzovat_vsetko = st.checkbox("Analyzovať celé dostupné obdobie", value=True)

    # Logika pre dátumy
    if analyzovat_vsetko:
        start_date = db_min_date
        end_date = db_max_date
    else:
        if len(vybrany_datum) == 2:
            start_date, end_date = vybrany_datum
        else:
            start_date = end_date = vybrany_datum[0]

    str_start_date = start_date.strftime("%Y-%m-%d")
    str_end_date = end_date.strftime("%Y-%m-%d")

    st.write("") 
    
    with st.spinner("Pripravujem AI model..."):
        sentiment_model = load_nlp_tools()

    # --- TLAČIDLO SPUSTENIA ---
    if st.button("🚀 Spustiť Analýzu a Zobraziť Dashboard", type="primary", use_container_width=True):
        if not vybrane_pobocky:
            st.error("⚠️ Vyberte aspoň jednu pobočku!")
            return

        st.markdown("---")
        
        placeholders = ', '.join(['?'] * len(vybrane_pobocky))
        
        # === KROK 1: DELTA LOAD ===
        with st.spinner("Hľadám nové recenzie pre zvolené pobočky a obdobie..."):
            with sqlite3.connect(db_path) as conn:
                query_new = f"""
                SELECT r.review_id, r.text
                FROM raw_reviews r
                LEFT JOIN processed_sentiment p ON r.review_id = p.review_id
                WHERE p.review_id IS NULL 
                  AND r.text IS NOT NULL AND r.text != ''
                  AND r.pobocka IN ({placeholders})
                  AND r.date BETWEEN ? AND ?
                """
                sql_params = tuple(vybrane_pobocky) + (str_start_date, str_end_date)
                
                df_new = pd.read_sql_query(query_new, conn, params=sql_params)
                
                if not df_new.empty:
                    st.info(f"💡 Našiel som {len(df_new)} nových recenzií! Púšťam AI motor...")
                    cursor = conn.cursor()
                    progress_bar = st.progress(0)
                    
                    for i, row in df_new.iterrows():
                        r_id = row['review_id']
                        text = row['text']
                        
                        raw_sent = sentiment_model(text)[0]
                        sent_data = process_sentiment_results(raw_sent)
                        cursor.execute("INSERT INTO processed_sentiment (review_id, sentiment_label, sentiment_score) VALUES (?, ?, ?)", 
                                       (r_id, sent_data['label'], sent_data['max_score']))
                        
                        aspects = extract_aspects_ultimate(text, sentiment_model)
                        for asp in aspects:
                            cursor.execute("INSERT INTO aspect_analysis (review_id, aspekt, veta, sentiment) VALUES (?, ?, ?, ?)", 
                                           (r_id, asp['Aspekt'], asp['Časť vety'], asp['Zistený stav']))
                        
                        progress_bar.progress((i + 1) / len(df_new))
                    
                    conn.commit()
                    st.success("✅ Nové dáta boli úspešne analyzované a uložené do databázy.")
                else:
                    st.success("✨ Tento výber je už plne zanalyzovaný. Načítavam report z databázy...")

        # === KROK 2: NAČÍTANIE ULOŽENÝCH DÁT ===
        with sqlite3.connect(db_path) as conn:
            query_overall = f"""
            SELECT r.pobocka, r.text, p.sentiment_label as Sentiment
            FROM raw_reviews r
            JOIN processed_sentiment p ON r.review_id = p.review_id
            WHERE r.pobocka IN ({placeholders})
              AND r.date BETWEEN ? AND ?
            """
            df_overall_data = pd.read_sql_query(query_overall, conn, params=sql_params)
            
            query_aspects = f"""
            SELECT r.pobocka, a.aspekt as Aspekt, a.sentiment as Sentiment
            FROM raw_reviews r
            JOIN aspect_analysis a ON r.review_id = a.review_id
            WHERE r.pobocka IN ({placeholders})
              AND r.date BETWEEN ? AND ?
            """
            df_aspect_data = pd.read_sql_query(query_aspects, conn, params=sql_params)

        if df_overall_data.empty:
            st.warning("⚠️ V databáze sa pre tento výber (zvolené pobočky a dátum) nenachádzajú žiadne spracované dáta.")
            return

        # === KROK 3: VYKRESLENIE DASHBOARDU ===
        pobocky_na_vykreslenie = df_overall_data['pobocka'].unique()
        color_map = {'Pozitívny': '#2ecc71', 'Neutrálny': '#f1c40f', 'Negatívny': '#e74c3c'}

        for pobocka in pobocky_na_vykreslenie:
            st.subheader(f"📍 Pobočka: {pobocka}")
            
            df_branch_overall = df_overall_data[df_overall_data['pobocka'] == pobocka]
            df_branch_aspects = df_aspect_data[df_aspect_data['pobocka'] == pobocka]
            
            sent_counts = df_branch_overall['Sentiment'].value_counts().reset_index()
            sent_counts.columns = ['Sentiment', 'Počet']
            
            fig_pie = px.pie(sent_counts, values='Počet', names='Sentiment', hole=0.4, title="Celkový sentiment pobočky", color='Sentiment', color_discrete_map=color_map) if not sent_counts.empty else None
            if fig_pie: fig_pie.update_layout(margin=dict(t=40, b=0, l=0, r=0), height=300)

            fig_bar = None
            if not df_branch_aspects.empty:
                asp_counts = df_branch_aspects.groupby(['Aspekt', 'Sentiment']).size().reset_index(name='Počet')
                fig_bar = px.bar(asp_counts, x="Aspekt", y="Počet", color="Sentiment", title="Analýza konkrétnych aspektov (ABSA)", color_discrete_map=color_map, barmode='group')
                fig_bar.update_layout(margin=dict(t=40, b=0, l=0, r=0), height=350)

            col1, col2 = st.columns([1, 1.5]) 
            with col1:
                if fig_pie: st.plotly_chart(fig_pie, use_container_width=True, key=f"pie_{pobocka}")
            with col2:
                if fig_bar: st.plotly_chart(fig_bar, use_container_width=True, key=f"bar_{pobocka}")
                else: st.info("Žiadne zachytené aspekty pre túto pobočku.")
            
            with st.expander(f"Zobraziť recenzie pobočky - {pobocka} ({len(df_branch_overall)} záznamov)"):
                st.dataframe(df_branch_overall[['text', 'Sentiment']], use_container_width=True)

            st.divider()
        
        # --- KROK 4: TABUĽKY NA KONCI ---
        st.markdown("---")
        draw_summary_tables(df_overall_data, df_aspect_data)

if __name__ == "__main__":
    main()