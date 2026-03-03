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
from streamlit_option_menu import option_menu 

# ==========================================
# ZÁKLADNÉ NASTAVENIA A ZDIEĽANÉ FUNKCIE
# ==========================================
st.set_page_config(
    page_title="Sentiment & Aspektová Analýza",
    page_icon="📊",
    layout="wide"
)

# Načítanie modelu do cache, aby bežal len raz
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

def lemmatize_text(text):
    words = word_tokenize(text, language='slovene')
    lemmatized_words = [simplemma.lemmatize(word, lang='sk') for word in words]
    return " ".join(lemmatized_words).lower()

# TVOJ UPRAVENÝ LEXIKÓN
ASPEKTOVE_SADY = {
    'Personál': ['personál', 'obsluha', 'predavač', 'predavačka', 'zamestnanec', 'poradca', 'pokladňa', 'prístup', 'ochota', 'poradiť', 'rad'],
    'Cena': ['cena', 'drahý', 'lacný', 'zľava', 'výpredaj', 'akcia', 'pomer', 'kvalita', 'peniaz', 'účet'],
    'E-shop': ['doprava', 'doručenie', 'kuriér', 'rýchlo', 'neskoro', 'balík', 'dodanie', 'eshop', 'objednávka', 'sklad', 'vyzdvihnutie', 'stránka', 'dostupnosť'],
    'Predajňa a Prostredie': ['prostredie', 'predajňa', 'obchod', 'pobočka', 'čistota', 'atmosféra', 'miesto', 'parkovanie', 'kabínka', 'výber', 'sortiment'],
    'Produkty': ['produkt', 'tovar', 'výrobok', 'materiál', 'oblečenie', 'obuv', 'topánka', 'vybavenie', 'pokazený', 'reklamácia', 'vydržať', 'rozpadnúť', 'strih', 'číslo', 'bicykel', 'stan', 'bunda', 'tričko', 'veľkosť']
}

def process_sentiment_results(results):
    scores = {"Pozitívny": 0.0, "Neutrálny": 0.0, "Negatívny": 0.0}
    label_map = {"positive": "Pozitívny", "label_2": "Pozitívny", "neutral": "Neutrálny", "label_1": "Neutrálny", "negative": "Negatívny", "label_0": "Negatívny"}
    for res in results:
        label = res['label'].lower()
        slovak_label = label_map.get(label, label)
        if slovak_label in scores:
            scores[slovak_label] = res['score']
    polarity = scores["Pozitívny"] - scores["Negatívny"]
    max_label = max(scores, key=scores.get)
    return {"scores": scores, "label": max_label, "max_score": scores[max_label], "polarity": polarity}

def extract_aspects_ultimate(text, model):
    found_aspects = []
    debug_info = []
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
                matched_word_info = ""
                for keyword in keywords:
                    keyword_norm = unidecode(keyword)
                    if keyword_norm in clause_normalized:
                        aspect_found = True
                        matched_word_info = f"Presná zhoda: '{keyword_norm}'"
                        break
                    for word in clause_words:
                        similarity = fuzz.ratio(keyword_norm, word)
                        if similarity >= 80:
                            aspect_found = True
                            matched_word_info = f"Fuzzy zhoda ({similarity}%): '{word}' ≈ '{keyword_norm}'"
                            break
                    if aspect_found: break
                if aspect_found:
                    raw_result = model(clause)[0]
                    sentiment_data = process_sentiment_results(raw_result)
                    debug_info.append({"Časť vety": clause, "Nájdené cez": matched_word_info, "Priradený aspekt": aspect_name})
                    found_aspects.append({"Aspekt": aspect_name, "Časť vety": clause, "Zistený stav": sentiment_data['label'], "Polarita": sentiment_data['polarity']})
    return found_aspects, debug_info

# --- NOVÁ FUNKCIA: JEDNODUCHÉ TABUĽKY ---
# ==========================================
# NOVÁ FUNKCIA: TABUĽKY S GRAFICKÝMI UKAZOVATEĽMI
# ==========================================
def draw_summary_tables(df_overall, df_aspects):
    st.markdown("##  Manažérsky Súhrn")
    st.markdown("Prehľadné tabuľky s výsledkami.")

    # 1. KPI TABUĽKA (s Progress Barmi)
    st.subheader("Celková úspešnosť pobočiek")
    
    # Výpočet percent
    summary = df_overall.groupby('pobocka')['Sentiment'].value_counts(normalize=True).unstack().fillna(0) * 100
    summary = summary.rename(columns={'Pozitívny': 'Pozitívne %', 'Negatívny': 'Negatívne %', 'Neutrálny': 'Neutrálne %'})
    
    # Pridanie počtu recenzií
    counts = df_overall['pobocka'].value_counts()
    summary['Počet hodnotení'] = counts
    
    # Zoradenie
    cols = ['Počet hodnotení', 'Pozitívne %', 'Neutrálne %', 'Negatívne %']
    existing_cols = [c for c in cols if c in summary.columns]
    summary = summary[existing_cols].sort_values(by='Pozitívne %', ascending=False)

    # Konfigurácia stĺpcov pre Streamlit
    column_config = {
        "Počet hodnotení": st.column_config.NumberColumn(
            "Počet hodnotení",
            help="Celkový počet hodnotení",
            format="%d 👤",  # Pridá ikonku panáčika
        ),
        "Pozitívne %": st.column_config.ProgressColumn(
            "Pozitívne %",
            help="Percento spokojných zákazníkov",
            format="%.1f%%",
            min_value=0,
            max_value=100,
        ),
        "Negatívne %": st.column_config.ProgressColumn(
            "Negatívne %",
            help="Percento nespokojných zákazníkov",
            format="%.1f%%",
            min_value=0,
            max_value=100,
        ),
        "Neutrálne %": st.column_config.NumberColumn(
            "Neutrálne %",
            format="%.1f%%"
        )
    }

    # Vykreslenie
    st.dataframe(
        summary,
        column_config=column_config,
        use_container_width=True
    )

    # 2. ASPEKTOVÁ MATICA 
    if not df_aspects.empty:
        st.subheader("Detailný súhrn spokojnosti podľa aspektov")
        st.markdown("Tabuľka ukazuje **% hodnotení pre daný aspekt**")

        aspect_totals = df_aspects.groupby(['pobocka', 'Aspekt']).size()
        aspect_positives = df_aspects[df_aspects['Sentiment'] == 'Pozitívny'].groupby(['pobocka', 'Aspekt']).size()
        
        aspect_matrix = (aspect_positives / aspect_totals * 100).unstack()
        aspect_matrix = aspect_matrix.fillna(0)
        
        # Toto zvýrazní najvyššie číslo v riadku (bez potreby matplotlibu)
        st.dataframe(
            aspect_matrix.style.highlight_max(axis=1, color='#D1E7DD', props='font-weight:bold;').format("{:.0f}%"),
            use_container_width=True
        )
    else:
        st.info("Pre zvolené kritériá neboli nájdené žiadne detailné aspekty.")


# ==========================================
# MODUL 1: ANALÝZA POBOČIEK (DASHBOARD)
# ==========================================
def run_dashboard_module(sentiment_model, db_path):
    st.title("📊 Sentiment porovnávač pobočiek")
    st.markdown("Nastavte si filtre a porovnajte sentiment pobočiek.")
    
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

    if not vsetky_pobocky:
        st.warning("⚠️ Databáza je prázdna, nie sú k dispozícii žiadne hodnotenia.")
        return

    col_filter1, col_filter2 = st.columns(2)
    with col_filter1:
        vybrane_pobocky = st.multiselect("📍 Vyberte pobočky na analýzu:", options=vsetky_pobocky)
    with col_filter2:
        vybrany_datum = st.date_input("📅 Vyberte obdobie (Od - Do):", value=(db_min_date, db_max_date))
        analyzovat_vsetko = st.checkbox("Analyzovať celé dostupné obdobie", value=True)

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
    
    if st.button("Spustiť Analýzu", type="primary", use_container_width=True):
        if not vybrane_pobocky:
            st.error("⚠️ Vyberte aspoň jednu pobočku!")
            return

        st.markdown("---")
        placeholders = ', '.join(['?'] * len(vybrane_pobocky))
        
        with st.spinner("Hľadám nové hodnotenia pre zvolené pobočky a obdobie..."):
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
                    st.info(f"💡 Bolo nájdených {len(df_new)} nových hodnotení! Spúšťam analýzu.")
                    cursor = conn.cursor()
                    progress_bar = st.progress(0)
                    
                    for i, row in df_new.iterrows():
                        r_id = row['review_id']
                        text = row['text']
                        
                        raw_sent = sentiment_model(text)[0]
                        sent_data = process_sentiment_results(raw_sent)
                        cursor.execute("INSERT INTO processed_sentiment (review_id, sentiment_label, sentiment_score) VALUES (?, ?, ?)", 
                                       (r_id, sent_data['label'], sent_data['max_score']))
                        
                        aspects, _ = extract_aspects_ultimate(text, sentiment_model)
                        for asp in aspects:
                            cursor.execute("INSERT INTO aspect_analysis (review_id, aspekt, veta, sentiment) VALUES (?, ?, ?, ?)", 
                                           (r_id, asp['Aspekt'], asp['Časť vety'], asp['Zistený stav']))
                        
                        progress_bar.progress((i + 1) / len(df_new))
                    
                    conn.commit()
                    st.success("✅ Nové dáta boli úspešne analyzované a uložené do databázy.")
                else:
                    st.success("✨ Tento výber je už plne zanalyzovaný. Načítavam dáta z databázy.")

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
            
            with st.expander(f"Zobraziť hodnotenia pobočky - {pobocka} ({len(df_branch_overall)} záznamov)"):
                st.dataframe(df_branch_overall[['text', 'Sentiment']], use_container_width=True)
            st.divider()
        
        # --- TABUĽKY NA KONCI (PRIDANÉ) ---
        st.markdown("---")
        draw_summary_tables(df_overall_data, df_aspect_data)


# ==========================================
# MODUL 2: DETAILNÁ ANALÝZA 1 RECENZIE (LAB)
# ==========================================
def run_laboratory_module(sentiment_model):
    st.title("Detailná Analýza")

    user_text = st.text_area(
        "Vložte text, ktorý chcete analyzovať:", 
        height=120, 
        value=""
    )

    if st.button("Analyzovať text", type="primary"):
        if not user_text.strip():
            st.warning("Zadajte text na analýzu.")
            return

        st.markdown("---")
        
        # 1. CELKOVÝ SENTIMENT
        st.subheader("Celkový Sentiment Textu")
        raw_sent_results = sentiment_model(user_text)[0]
        sent_data = process_sentiment_results(raw_sent_results)
        
        col1, col2 = st.columns([1, 2])
        with col1:
            if sent_data['label'] == "Pozitívny": st.success(f"**Dominantný sentiment:** {sent_data['label']}")
            elif sent_data['label'] == "Negatívny": st.error(f"**Dominantný sentiment:** {sent_data['label']}")
            else: st.info(f"**Dominantný sentiment:** {sent_data['label']}")
            st.metric(label="Vypočítaná Polarita (-1 až +1)", value=f"{sent_data['polarity']:.2f}")
        
        with col2:
            st.progress(sent_data['scores']['Pozitívny'], text=f"Pozitívny ({sent_data['scores']['Pozitívny']:.1%})")
            st.progress(sent_data['scores']['Neutrálny'], text=f"Neutrálny ({sent_data['scores']['Neutrálny']:.1%})")
            st.progress(sent_data['scores']['Negatívny'], text=f"Negatívny ({sent_data['scores']['Negatívny']:.1%})")

        st.markdown("---")

        # 2. ASPEKTY
        st.subheader("Detailný rozbor nájdených aspektov")
        with st.spinner("Hľadám aspekty cez Fuzzy Matching..."):
            extracted_aspects, debug_info = extract_aspects_ultimate(user_text, sentiment_model)
            
        if not extracted_aspects:
            st.warning("V texte neboli nájdené žiadne kľúčové slová z vášho lexikónu.")
        else:
            for item in extracted_aspects:
                status = item['Zistený stav']
                if status == "Pozitívny": color, emoji = "green", "🟢"
                elif status == "Negatívny": color, emoji = "red", "🔴"
                else: color, emoji = "orange", "🟡"
                
                st.markdown(f"**Aspekt: `{item['Aspekt']}`**")
                st.markdown(f"> *\"{item['Časť vety']}\"*")
                st.markdown(f"**{emoji} {status}** (Skóre polarity: <span style='color:{color}'>{item['Polarita']:.2f}</span>)", unsafe_allow_html=True)
                st.markdown("---")


# ==========================================
# MODUL 3: DATA WAREHOUSE
# ==========================================
def run_data_warehouse_module(db_path):
    st.title("🗄️ Data Warehouse")
    st.markdown("Prehľad uložených dát. Tabuľky si môžete prezerať a exportovať.")

    with sqlite3.connect(db_path) as conn:
        try:
            raw_df = pd.read_sql_query("SELECT * FROM raw_reviews", conn)
            sent_df = pd.read_sql_query("SELECT * FROM processed_sentiment", conn)
            asp_df = pd.read_sql_query("SELECT * FROM aspect_analysis", conn)
        except Exception as e:
            st.error(f"❌ Chyba pri načítaní databázy: {e}")
            return

    st.subheader("Metriky Databázy")
    col1, col2, col3 = st.columns(3)
    col1.metric(label="Online hodnotenia", value=len(raw_df))
    col2.metric(label="Spracovaný sentiment", value=len(sent_df))
    col3.metric(label="Nájdené aspekty", value=len(asp_df))
    st.divider()

    tab1, tab2, tab3 = st.tabs(["Online hodnotenia", "Sentiment", "Aspekty"])

    with tab1:
        st.dataframe(raw_df, use_container_width=True, height=400)
        st.download_button("Stiahnuť raw_reviews (.csv)", raw_df.to_csv(index=False).encode('utf-8'), 'raw_reviews.csv', 'text/csv')
    with tab2:
        st.dataframe(sent_df, use_container_width=True, height=400)
        st.download_button("Stiahnuť processed_sentiment (.csv)", sent_df.to_csv(index=False).encode('utf-8'), 'processed_sentiment.csv', 'text/csv')
    with tab3:
        st.dataframe(asp_df, use_container_width=True, height=400)
        st.download_button("Stiahnuť aspect_analysis (.csv)", asp_df.to_csv(index=False).encode('utf-8'), 'aspect_analysis.csv', 'text/csv')


# ==========================================
# MASTER ROUTER (HORNÉ MENU)
# ==========================================
def main():
    # Inicializácia modelu (stane sa len raz vďaka cache)
    with st.spinner("Systém inicializuje jazykové a AI modely..."):
        sentiment_model = load_nlp_tools()
        
    db_path = 'decathlon_warehouse.db'

    # Vytvorenie elegantného horného menu
    selected_module = option_menu(
        menu_title=None,  # Skryje nadpis menu
        options=["Analýza Pobočiek", "Detailná Analýza", "Data Warehouse"],
        icons=["bar-chart-line", "microscope", "database"], 
        menu_icon="cast",
        default_index=0,
        orientation="horizontal",
        styles={
            "container": {"padding": "0!important", "background-color": "#fafafa", "margin-bottom": "20px"},
            "icon": {"color": "orange", "font-size": "18px"}, 
            "nav-link": {"font-size": "16px", "text-align": "center", "margin":"0px", "--hover-color": "#eee"},
            "nav-link-selected": {"background-color": "#0047AB", "color": "white"}, # Decathlon modrá
        }
    )

    # Smerovanie
    if selected_module == "Analýza Pobočiek":
        run_dashboard_module(sentiment_model, db_path)
    elif selected_module == "Detailná Analýza":
        run_laboratory_module(sentiment_model)
    elif selected_module == "Data Warehouse":
        run_data_warehouse_module(db_path)

if __name__ == "__main__":
    main()