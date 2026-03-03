import streamlit as st
import pandas as pd
import re
from transformers import pipeline
from unidecode import unidecode
from simplemma import lemmatize
from thefuzz import fuzz
import nltk
from nltk.tokenize import sent_tokenize

# --- INICIALIZÁCIA ---
st.set_page_config(page_title="Decathlon NLP Emotion Engine (Google model)", layout="wide")

@st.cache_resource
def load_models():
    # 1. Model na Sentiment (XLM-RoBERTa Base)
    sent_pipe = pipeline("sentiment-analysis", model="cardiffnlp/twitter-xlm-roberta-base-sentiment")
    
    # 2. Univerzálny Google/Facebook model (XLM-RoBERTa Large)
    # Tento model je Zero-Shot, čo znamená, že mu povieme emócie priamo v slovenčine!
    emot_pipe = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    
    return sent_pipe, emot_pipe

sentiment_pipeline, emotion_pipeline = load_models()

# Stiahnutie dát pre rozdelenie viet
nltk.download('punkt', quiet=True)

# --- KONFIGURÁCIA ---
CONJUNCTIONS = r'\b(ale|no|avsak|hoci|alebo|pricom|lenze)\b'

# Definujeme emócie, ktoré má model v texte hľadať (v slovenčine)
ZOZNAM_EMOCII = ["radosť", "hnev", "smútok", "strach", "prekvapenie", "nadšenie"]

ASPEKTOVE_SADY = {
    'Personál a Obsluha': ['personal', 'obsluha', 'predavac', 'predavacka', 'zamestnanec', 'poradca', 'pokladna', 'pristup', 'ochota', 'poradit'],
    'Cena a Hodnota': ['cena', 'drahy', 'lacny', 'zlava', 'vypredaj', 'akcia', 'pomer', 'peniaz'],
    'Dostupnosť a E-shop': ['doprava', 'dorucenie', 'kurier', 'rychlo', 'neskoro', 'balik', 'dodanie', 'eshop', 'objednavka', 'sklad', 'vyzdvihnutie'],
    'Predajňa a Prostredie': ['prostredie', 'predajna', 'obchod', 'pobocka', 'cistota', 'atmosfera', 'miesto', 'parkovanie', 'kabinka', 'vyber', 'sortiment'],
    'Kvalita Produktov': ['produkt', 'tovar', 'vyrobok', 'material', 'oblecenie', 'obuv', 'topanka', 'vybavenie', 'pokazeny', 'reklamacia', 'vydrzat', 'rozpadnut', 'strih', 'bicykel', 'stan', 'bunda', 'tricko', 'velkost']
}

# --- FUNKCIE ---
def extract_aspects_ultimate(text, threshold=80):
    sentences = sent_tokenize(text)
    all_clauses = []
    
    for sent in sentences:
        parts = re.split(CONJUNCTIONS, sent, flags=re.IGNORECASE)
        all_clauses.extend([p.strip() for p in parts if len(p.strip()) > 2])

    results = []
    
    for clause in all_clauses:
        clean_clause = unidecode(clause.lower())
        words = clean_clause.split()
        
        found_aspect = None
        matched_word = None
        
        # Hľadanie aspektu
        for word in words:
            lemma = lemmatize(word, lang='sk')
            for aspect_name, keywords in ASPEKTOVE_SADY.items():
                for kw in keywords:
                    score = fuzz.ratio(lemma, kw)
                    if score >= threshold:
                        found_aspect = aspect_name
                        matched_word = word
                        break
                if found_aspect: break
            if found_aspect: break
        
        if found_aspect:
            # 1. Sentiment (Klasika)
            sent_res = sentiment_pipeline(clause)[0]
            label_map = {'positive': 'Pozitívny 🟢', 'neutral': 'Neutrálny 🟡', 'negative': 'Negatívny 🔴'}
            
            # 2. Emócia (Zero-Shot klasifikácia)
            # Modelu pošleme text a zoznam emócií, on vyberie tú najpravdepodobnejšiu
            emot_res = emotion_pipeline(clause, candidate_labels=ZOZNAM_EMOCII)
            top_emocia = emot_res['labels'][0]
            emot_score = emot_res['scores'][0]
            
            # Pridanie emoji k výsledku pre lepšiu vizualizáciu
            emoji_map = {"radosť": "Radosť 😃", "hnev": "Hnev 😡", "smútok": "Smútok 😢", 
                         "strach": "Strach 😨", "prekvapenie": "Prekvapenie 😮", "nadšenie": "Nadšenie ❤️"}
            
            results.append({
                'Aspekt': found_aspect,
                'Kľúčové slovo': matched_word,
                'Sentiment': label_map.get(sent_res['label'], sent_res['label']),
                'Emócia': emoji_map.get(top_emocia, top_emocia),
                'Časť vety': clause,
                'Istota emócie': round(emot_score, 2)
            })
            
    return results

# --- UI (STREAMLIT) ---
st.title("🚀 Decathlon NLP: Ultimate Emotion & Aspect Engine")
st.info("Tento modul používa model Facebook BART-Large pre Zero-Shot klasifikáciu emócií.")

input_text = st.text_area("Vložte recenziu:", height=150, placeholder="Sem skopírujte text recenzie...")

if st.button("Spustiť hĺbkovú analýzu"):
    if input_text:
        with st.spinner('Analýza prebieha (používame Large Transformer modely)...'):
            data = extract_aspects_ultimate(input_text)
        
        if data:
            st.subheader("Výsledky")
            df = pd.DataFrame(data)
            # Zobrazíme tabuľku bez stĺpca "Časť vety" (ten dáme do detailu)
            st.dataframe(df.drop(columns=['Časť vety']), use_container_width=True)
            
            st.subheader("Detailný rozbor viet")
            for res in data:
                with st.expander(f"Aspekt: {res['Aspekt']} | {res['Emócia']}"):
                    st.write(f"**Analyzovaný text:** {res['Časť vety']}")
                    st.write(f"**Identifikovaný sentiment:** {res['Sentiment']}")
                    st.progress(res['Istota emócie'])
        else:
            st.warning("V texte nebol nájdený žiadny sledovaný aspekt.")
    else:
        st.error("Zadajte text recenzie!")