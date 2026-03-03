import streamlit as st
from transformers import pipeline
import pandas as pd

# --- 1. Nastavenie stránky ---
st.set_page_config(
    page_title="Detailná Analýza Recenzie",
    page_icon="🔬",
    layout="wide"
)

# --- 2. Načítanie modelov (Kešované pre rýchlosť) ---
@st.cache_resource
def load_sentiment_model():
    """Načíta model pre celkový sentiment."""
    model_path = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
    # top_k=None vráti pravdepodobnosti pre všetky 3 kategórie
    return pipeline("sentiment-analysis", model=model_path, tokenizer=model_path, top_k=None)

@st.cache_resource
def load_zeroshot_model():
    """Načíta NLI model pre Aspektovú analýzu (ABSA)."""
    model_path = "symanto/xlm-roberta-base-snli-mnli-anli-xnli"
    return pipeline("zero-shot-classification", model=model_path)

# --- 3. Pomocné funkcie ---
def process_overall_sentiment(results):
    """Spracuje surové výsledky z cardiffnlp modelu do pekného formátu."""
    scores = {"Pozitívny": 0.0, "Neutrálny": 0.0, "Negatívny": 0.0}
    
    # Mapovanie štítkov na slovenčinu
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

# --- 4. Hlavné používateľské rozhranie ---
def main():
    st.title("🔬 Laboratórium: Detailná analýza jednej recenzie")
    st.markdown("""
    Tento modul využíva **XLM-RoBERTa** pre celkový sentiment a **Zero-Shot NLI v dvoch krokoch**, 
    aby sme predišli halucináciám (priraďovaniu sentimentu k aspektom, ktoré sa v texte nenachádzajú).
    """)

    # Načítanie modelov
    with st.spinner("Pripravujem neurónové siete... (Prvé načítanie môže trvať dlhšie)"):
        sentiment_pipe = load_sentiment_model()
        zeroshot_pipe = load_zeroshot_model()

    # Vstupy od používateľa
    user_text = st.text_area(
        "Vložte text recenzie:", 
        height=120, 
        # Schválne slovo "zamestnankyňa" a nespomenuté parkovanie/prostredie
        value="Jedlo bolo fantastické, ale zamestnankyňa bola veľmi pomalá a drzá. Ceny ujdú."
    )
    
    # Preddefinované aspekty
    default_aspects = ["Jedlo", "Obsluha", "Cena", "Prostredie", "Parkovanie"]
    selected_aspects = st.multiselect(
        "Vyberte aspekty, ktoré chcete v texte hľadať:", 
        options=default_aspects, 
        default=["Jedlo", "Obsluha", "Cena", "Prostredie", "Parkovanie"]
    )

    if st.button("Analyzovať text", type="primary"):
        if not user_text.strip():
            st.warning("Zadajte text na analýzu.")
            return

        st.markdown("---")
        
        # === ČASŤ 1: CELKOVÝ SENTIMENT ===
        st.subheader("1. Celkový Sentiment Textu")
        
        # Pipeline vráti list of lists, zoberieme prvý
        raw_sent_results = sentiment_pipe(user_text)[0]
        sent_data = process_overall_sentiment(raw_sent_results)
        
        col1, col2 = st.columns([1, 2])
        with col1:
            # Farebné odlíšenie celkového labelu
            if sent_data['label'] == "Pozitívny":
                st.success(f"**Dominantný sentiment:** {sent_data['label']}")
            elif sent_data['label'] == "Negatívny":
                st.error(f"**Dominantný sentiment:** {sent_data['label']}")
            else:
                st.info(f"**Dominantný sentiment:** {sent_data['label']}")
                
            st.metric(label="Vypočítaná Polarita", value=f"{sent_data['polarity']:.2f}", 
                      help="Rozdiel medzi pravdepodobnosťou Pozitívneho a Negatívneho sentimentu (škála -1 až +1).")
        
        with col2:
            st.markdown("**Pravdepodobnosti modelu:**")
            st.progress(sent_data['scores']['Pozitívny'], text=f"Pozitívny ({sent_data['scores']['Pozitívny']:.1%})")
            st.progress(sent_data['scores']['Neutrálny'], text=f"Neutrálny ({sent_data['scores']['Neutrálny']:.1%})")
            st.progress(sent_data['scores']['Negatívny'], text=f"Negatívny ({sent_data['scores']['Negatívny']:.1%})")

        st.markdown("---")

        # === ČASŤ 2: ZERO-SHOT ABSA (DVOJKROKOVÁ OCHRANA) ===
        st.subheader("2. Inteligentná Aspektová Analýza (Zero-Shot)")
        
        results_absa = []
        
        with st.spinner("Skúmam jednotlivé aspekty logickým dvojkrokovým overením..."):
            for aspect in selected_aspects:
                
                # KROK 1: Detekcia prítomnosti (Je tento aspekt vôbec v texte?)
                presence_labels = ["mentioned", "not mentioned"]
                presence_template = f"The topic of '{aspect}' is {{}} in this text."
                
                presence_output = zeroshot_pipe(
                    user_text, 
                    presence_labels, 
                    hypothesis_template=presence_template,
                    multi_label=False
                )
                
                # Zistíme, či model vybral "mentioned" ako najpravdepodobnejšie
                is_mentioned = presence_output['labels'][0] == "mentioned"
                
                if not is_mentioned:
                    # Ak sa nespomína, rovno to zapíšeme a ideme na ďalší aspekt (preskočíme sentiment)
                    results_absa.append({
                        "Aspekt": aspect,
                        "Zistený stav": "Nespomína sa",
                        "Istota modelu": presence_output['scores'][0]
                    })
                    continue 
                
                # KROK 2: Ak sa aspekt spomína, aká je jeho emócia?
                sentiment_labels = ["positive", "negative"]
                sentiment_template = f"Regarding the aspect '{aspect}', the sentiment is {{}}."
                
                sent_output = zeroshot_pipe(
                    user_text, 
                    sentiment_labels, 
                    hypothesis_template=sentiment_template,
                    multi_label=False
                )
                
                top_label = "Pozitívny" if sent_output['labels'][0] == "positive" else "Negatívny"
                top_score = sent_output['scores'][0]
                
                results_absa.append({
                    "Aspekt": aspect,
                    "Zistený stav": top_label,
                    "Istota modelu": top_score
                })

        # Zobrazenie výsledkov ABSA
        cols = st.columns(len(selected_aspects))
        
        for i, res in enumerate(results_absa):
            with cols[i % len(cols)]:
                status = res["Zistený stav"]
                if status == "Pozitívny":
                    st.success(f"**{res['Aspekt']}**\n\n🟢 Pozitívny\n\n*(Istota: {res['Istota modelu']:.0%})*")
                elif status == "Negatívny":
                    st.error(f"**{res['Aspekt']}**\n\n🔴 Negatívny\n\n*(Istota: {res['Istota modelu']:.0%})*")
                else:
                    st.info(f"**{res['Aspekt']}**\n\n⚪ Nespomína sa\n\n*(Istota: {res['Istota modelu']:.0%})*")

if __name__ == "__main__":
    main()