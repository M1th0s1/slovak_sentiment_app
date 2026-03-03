import streamlit as st
from transformers import pipeline
import re
import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import simplemma
from unidecode import unidecode
from thefuzz import fuzz

# --- 1. Nastavenie stránky ---
st.set_page_config(
    page_title="Detailná Analýza Recenzie",
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

# --- 4. Zjednodušený Lexikón ---
# --- 4. Špecializovaný Lexikón pre DECATHLON ---
# --- 4. Zjednodušený Lexikón pre DECATHLON ---
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
    return {"scores": scores, "polarity": polarity, "label": max_label}

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
            
            # 1. Lematizácia
            lemmatized_clause = lemmatize_text(clause)
            
            # 2. Odstránenie diakritiky z textu klauzuly (pre porovnávanie)
            clause_normalized = unidecode(lemmatized_clause)
            clause_words = clause_normalized.split()
            
            # Hľadanie aspektov s Fuzzy Matchingom a Unidecode
            for aspect_name, keywords in ASPEKTOVE_SADY.items():
                aspect_found = False
                matched_word_info = ""
                
                for keyword in keywords:
                    # Odstránime diakritiku aj z kľúčového slova v slovníku
                    keyword_norm = unidecode(keyword)
                    
                    # Rýchly check na presnú zhodu (bez mäkčeňov)
                    if keyword_norm in clause_normalized:
                        aspect_found = True
                        matched_word_info = f"Presná zhoda: '{keyword_norm}'"
                        break
                        
                    # Ak nie je presná zhoda, hľadáme preklepy cez Fuzzy Matching
                    for word in clause_words:
                        similarity = fuzz.ratio(keyword_norm, word)
                        # Ak je zhoda aspoň 80% (napr. produktz vs produkt)
                        if similarity >= 80:
                            aspect_found = True
                            matched_word_info = f"Fuzzy zhoda ({similarity}%): '{word}' ≈ '{keyword_norm}'"
                            break
                            
                    if aspect_found:
                        break
                
                if aspect_found:
                    debug_info.append({
                        "Časť vety": clause,
                        "Nájdené cez": matched_word_info,
                        "Priradený aspekt": aspect_name
                    })
                    
                    # Vyhodnotenie sentimentu pôvodnej klauzuly
                    raw_result = model(clause)[0]
                    sentiment_data = process_sentiment_results(raw_result)
                    
                    found_aspects.append({
                        "Aspekt": aspect_name,
                        "Časť vety": clause,
                        "Zistený stav": sentiment_data['label'],
                        "Istota": sentiment_data['scores'][sentiment_data['label']],
                        "Polarita": sentiment_data['polarity']
                    })
                    
    return found_aspects, debug_info

# --- 6. Hlavné používateľské rozhranie ---
def main():
    st.title("🔬 Laboratórium: Ultimátna Analýza (Diakritika + Preklepy)")
    st.markdown("Tento modul si poradí s chýbajúcou diakritikou (`unidecode`) aj s preklepmi (`thefuzz`), takže nájde aj 'predavaci' alebo 'produktz'.")

    with st.spinner("Pripravujem AI model a jazykové nástroje..."):
        sentiment_model = load_nlp_tools()

    # Vstup používateľa s tvojimi chytákmi (bez diakritiky a s preklepom)
    user_text = st.text_area(
        "Vložte text recenzie:", 
        height=120, 
        value="produktz su fajn ale predavaci o nicom"
    )

    if st.button("Analyzovať text", type="primary"):
        if not user_text.strip():
            st.warning("Zadajte text na analýzu.")
            return

        st.markdown("---")
        
        # === ČASŤ 1: CELKOVÝ SENTIMENT ===
        st.subheader("1. Celkový Sentiment Textu")
        
        raw_sent_results = sentiment_model(user_text)[0]
        sent_data = process_sentiment_results(raw_sent_results)
        
        col1, col2 = st.columns([1, 2])
        with col1:
            if sent_data['label'] == "Pozitívny":
                st.success(f"**Dominantný sentiment:** {sent_data['label']}")
            elif sent_data['label'] == "Negatívny":
                st.error(f"**Dominantný sentiment:** {sent_data['label']}")
            else:
                st.info(f"**Dominantný sentiment:** {sent_data['label']}")
                
            st.metric(label="Vypočítaná Polarita (-1 až +1)", value=f"{sent_data['polarity']:.2f}")
        
        with col2:
            st.progress(sent_data['scores']['Pozitívny'], text=f"Pozitívny ({sent_data['scores']['Pozitívny']:.1%})")
            st.progress(sent_data['scores']['Neutrálny'], text=f"Neutrálny ({sent_data['scores']['Neutrálny']:.1%})")
            st.progress(sent_data['scores']['Negatívny'], text=f"Negatívny ({sent_data['scores']['Negatívny']:.1%})")

        st.markdown("---")

        # === ČASŤ 2: ASPEKTOVÁ ANALÝZA ===
        st.subheader("2. Detailný rozbor nájdených aspektov")
        
        with st.spinner("Hľadám aspekty cez Fuzzy Matching a analyzujem..."):
            extracted_aspects, debug_info = extract_aspects_ultimate(user_text, sentiment_model)
            
        if not extracted_aspects:
            st.warning("V texte neboli nájdené žiadne kľúčové slová z vášho lexikónu.")
        else:
            for item in extracted_aspects:
                status = item['Zistený stav']
                if status == "Pozitívny":
                    color = "green"
                    emoji = "🟢"
                elif status == "Negatívny":
                    color = "red"
                    emoji = "🔴"
                else:
                    color = "orange"
                    emoji = "🟡"
                
                st.markdown(f"**Aspekt: `{item['Aspekt']}`**")
                st.markdown(f"> *\"{item['Časť vety']}\"*")
                st.markdown(f"**{emoji} {status}** (Skóre polarity: <span style='color:{color}'>{item['Polarita']:.2f}</span>)", unsafe_allow_html=True)
                st.markdown("---")
                
        # === ČASŤ 3: LADIACI PANEL (DEBUG) ===
        with st.expander("🔍 Ladiaci panel (Ako algoritmus našiel preklepy?)"):
            st.markdown("Skontroluj si, ako algoritmus odhalil preklepy a chýbajúcu diakritiku:")
            if debug_info:
                st.table(pd.DataFrame(debug_info))
            else:
                st.info("Zatiaľ žiadne záznamy.")

if __name__ == "__main__":
    main()