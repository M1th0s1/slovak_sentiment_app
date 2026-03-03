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

# --- 1. Nastavenie stránky ---
st.set_page_config(
    page_title="Decathlon Analytics",
    page_icon="🔬",
    layout="wide"
)

# --- 2. Načítanie modelu a NLTK dát ---
@st.cache_resource
def load_nlp_tools():
    """Načíta model sentimentu a overí NLTK dáta."""
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
    """Prevedie všetky slová v texte na ich základný tvar (lemu)."""
    words = word_tokenize(text, language='slovene')
    lemmatized_words = [simplemma.lemmatize(word, lang='sk') for word in words]
    return " ".join(lemmatized_words).lower()

# --- 4. Špecializovaný Lexikón pre DECATHLON ---
ASPEKTOVE_SADY = {
    'Personál a Obsluha': [
        'personál', 'obsluha', 'predavač', 'predavačka', 'zamestnanec', 
        'poradca', 'pokladňa', 'prístup', 'ochota', 'poradiť', 'rad'
    ],
    'Cena a Hodnota': [
        'cena', 'drahý', 'lacný', 'zľava', 'výpredaj', 'akcia', 
        'pomer', 'kvalita', 'peniaz', 'účet'
    ],
    'Dostupnosť a E-shop': [
        'doprava', 'doručenie', 'kuriér', 'rýchlo', 'neskoro', 'balík', 
        'dodanie', 'eshop', 'objednávka', 'sklad', 'vyzdvihnutie', 'stránka', 'dostupnosť'
    ],
    'Predajňa a Prostredie': [
        'prostredie', 'predajňa', 'obchod', 'pobočka', 'čistota', 'atmosféra', 
        'miesto', 'parkovanie', 'kabínka', 'výber', 'sortiment'
    ],
    'Kvalita Produktov (Vybavenie a Oblečenie)': [
        'produkt', 'tovar', 'výrobok', 'materiál', 'oblečenie', 'obuv', 'topánka',
        'vybavenie', 'pokazený', 'reklamácia', 'vydržať', 'rozpadnúť', 'strih', 'číslo',
        'bicykel', 'stan', 'bunda', 'tričko', 'veľkosť' 
    ]
}

# --- 5. Analytické funkcie ---
def process_sentiment_results(results):
    scores = {"Pozitívny": 0.0, "Neutrálny": 0.0, "Negatívny": 0.0}
    label_map = {
        "positive": "Pozitívny", "label_2": "Pozitívny",
        "neutral": "Neutrálny",  "label_1": "Neutrálny",
        "negative": "Negatívny", "label_0": "Negatívny"
    }
    
    for res in results:
        label = res['label'].lower()
        slovak_label = label_map.get(label, label)
        if slovak_label in scores:
            scores[slovak_label] = res['score']
            
    polarity = scores["Pozitívny"] - scores["Negatívny"]
    max_label = max(scores, key=scores.get)
    return {"scores": scores, "polarity": polarity, "label": max_label, "max_score": scores[max_label]}

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
                            
                    if aspect_found:
                        break
                
                if aspect_found:
                    raw_result = model(clause)[0]
                    sentiment_data = process_sentiment_results(raw_result)
                    
                    found_aspects.append({
                        "Aspekt": aspect_name,
                        "Časť vety": clause,
                        "Zistený stav": sentiment_data['label']
                    })
                    
    return found_aspects

# --- 6. HLAVNÝ DASHBOARD ---
def main():
    st.title("📊 Decathlon Analytics: Analýza pobočiek")
    st.markdown("Systém inteligentne analyzuje len **nové** recenzie z databázy a aktualizuje reporty.")

    with st.spinner("Pripravujem AI model (Môže to chvíľku trvať)..."):
        sentiment_model = load_nlp_tools()

    if st.button("🚀 Spustiť Analýzu a Zobraziť Dashboard", type="primary", use_container_width=True):
        st.markdown("---")
        
        db_path = 'decathlon_warehouse.db'
        
        # === KROK 1: DELTA LOAD (Analýza iba nových dát) ===
        with st.spinner("Skontrolujem databázu a hľadám nové recenzie..."):
            with sqlite3.connect(db_path) as conn:
                # Nájdeme recenzie, ktoré sú v raw_reviews, ale ešte CHÝBAJÚ v processed_sentiment
                query_new = """
                SELECT r.review_id, r.text
                FROM raw_reviews r
                LEFT JOIN processed_sentiment p ON r.review_id = p.review_id
                WHERE p.review_id IS NULL AND r.text IS NOT NULL AND r.text != ''
                """
                df_new = pd.read_sql_query(query_new, conn)
                
                if not df_new.empty:
                    st.info(f"💡 Našiel som {len(df_new)} nových recenzií! Púšťam AI motor na ich spracovanie...")
                    
                    cursor = conn.cursor()
                    progress_bar = st.progress(0)
                    
                    for i, row in df_new.iterrows():
                        r_id = row['review_id']
                        text = row['text']
                        
                        # 1. Analýza a uloženie celkového sentimentu
                        raw_sent = sentiment_model(text)[0]
                        sent_data = process_sentiment_results(raw_sent)
                        
                        cursor.execute("""
                            INSERT INTO processed_sentiment (review_id, sentiment_label, sentiment_score)
                            VALUES (?, ?, ?)
                        """, (r_id, sent_data['label'], sent_data['max_score']))
                        
                        # 2. Analýza a uloženie aspektov (ABSA)
                        aspects = extract_aspects_ultimate(text, sentiment_model)
                        for asp in aspects:
                            cursor.execute("""
                                INSERT INTO aspect_analysis (review_id, aspekt, veta, sentiment)
                                VALUES (?, ?, ?, ?)
                            """, (r_id, asp['Aspekt'], asp['Časť vety'], asp['Zistený stav']))
                        
                        # Aktualizácia progress baru
                        progress_bar.progress((i + 1) / len(df_new))
                    
                    conn.commit() # ULOŽÍME VÝSLEDKY DO DATABÁZY!
                    st.success("✅ Nové dáta boli úspešne analyzované a bezpečne uložené do databázy.")
                else:
                    st.success("✨ Všetky recenzie sú už spracované. Generujem grafy z uložených dát...")

        # === KROK 2: NAČÍTANIE ULOŽENÝCH DÁT PRE GRAFY ===
        with sqlite3.connect(db_path) as conn:
            # Načítame spojené dáta pre grafy
            query_overall = """
            SELECT r.pobocka, r.text, p.sentiment_label as Sentiment
            FROM raw_reviews r
            JOIN processed_sentiment p ON r.review_id = p.review_id
            """
            df_overall_data = pd.read_sql_query(query_overall, conn)
            
            query_aspects = """
            SELECT r.pobocka, a.aspekt as Aspekt, a.sentiment as Sentiment
            FROM raw_reviews r
            JOIN aspect_analysis a ON r.review_id = a.review_id
            """
            df_aspect_data = pd.read_sql_query(query_aspects, conn)

        if df_overall_data.empty:
            st.warning("⚠️ Databáza je zatiaľ úplne prázdna.")
            return

        # === KROK 3: VYKRESLENIE DASHBOARDU ===
        pobocky = df_overall_data['pobocka'].unique()
        color_map = {'Pozitívny': '#2ecc71', 'Neutrálny': '#f1c40f', 'Negatívny': '#e74c3c'}

        for pobocka in pobocky:
            st.header(f"📍 Pobočka: {pobocka}")
            
            # Dáta len pre túto pobočku
            df_branch_overall = df_overall_data[df_overall_data['pobocka'] == pobocka]
            df_branch_aspects = df_aspect_data[df_aspect_data['pobocka'] == pobocka]
            
            # VIZUALIZÁCIA 1: Koláčový graf
            sent_counts = df_branch_overall['Sentiment'].value_counts().reset_index()
            sent_counts.columns = ['Sentiment', 'Počet']
            
            if not sent_counts.empty:
                fig_pie = px.pie(
                    sent_counts, values='Počet', names='Sentiment', hole=0.4,
                    title="Celkový sentiment pobočky",
                    color='Sentiment', color_discrete_map=color_map
                )
                fig_pie.update_layout(margin=dict(t=40, b=0, l=0, r=0), height=300)
            else:
                fig_pie = None

            # VIZUALIZÁCIA 2: Stĺpcový graf (ABSA)
            if not df_branch_aspects.empty:
                asp_counts = df_branch_aspects.groupby(['Aspekt', 'Sentiment']).size().reset_index(name='Počet')
                fig_bar = px.bar(
                    asp_counts, x="Aspekt", y="Počet", color="Sentiment",
                    title="Analýza konkrétnych aspektov (ABSA)",
                    color_discrete_map=color_map, barmode='group'
                )
                fig_bar.update_layout(margin=dict(t=40, b=0, l=0, r=0), height=350)
            else:
                fig_bar = None

            # Vykreslenie
            col1, col2 = st.columns([1, 1.5]) 
            with col1:
                if fig_pie:
                    st.plotly_chart(fig_pie, use_container_width=True, key=f"pie_{pobocka}")
            with col2:
                if fig_bar:
                    st.plotly_chart(fig_bar, use_container_width=True, key=f"bar_{pobocka}")
                else:
                    st.info("V recenziách neboli spomenuté žiadne špecifické aspekty.")
            
            # Surové dáta náhľad
            with st.expander(f"Zobraziť recenzie pobočky - {pobocka} ({len(df_branch_overall)} záznamov)"):
                st.dataframe(df_branch_overall[['text', 'Sentiment']], use_container_width=True)

            st.divider()

if __name__ == "__main__":
    main()