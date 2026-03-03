import streamlit as st
import pandas as pd
import sqlite3

# --- Nastavenie stránky ---
st.set_page_config(
    page_title="Data Warehouse",
    page_icon="🗄️",
    layout="wide"
)

def main():
    st.title("🗄️ Data Warehouse (Prehľad databázy)")
    st.markdown("Priamy prístup k uloženým dátam. Tabuľky si môžete prezerať, triediť kliknutím na hlavičku stĺpca alebo exportovať do Excelu (CSV).")

    db_path = 'decathlon_warehouse.db'

    # --- Načítanie dát z databázy ---
    with sqlite3.connect(db_path) as conn:
        try:
            raw_df = pd.read_sql_query("SELECT * FROM raw_reviews", conn)
            sent_df = pd.read_sql_query("SELECT * FROM processed_sentiment", conn)
            asp_df = pd.read_sql_query("SELECT * FROM aspect_analysis", conn)
        except Exception as e:
            st.error(f"❌ Chyba pri načítaní databázy: {e}")
            st.info("Uistite sa, že súbor databázy existuje a obsahuje potrebné tabuľky.")
            return

    # --- METRIKY (PREHĽAD) ---
    st.subheader("Rýchly prehľad")
    col1, col2, col3 = st.columns(3)
    col1.metric(label="Surové recenzie (Raw)", value=len(raw_df))
    col2.metric(label="Spracovaný sentiment", value=len(sent_df))
    col3.metric(label="Nájdené aspekty", value=len(asp_df))
    
    st.divider()

    # --- ZÁLOŽKY PRE JEDNOTLIVÉ TABUĽKY ---
    tab1, tab2, tab3 = st.tabs(["🥉 Bronzová vrstva (Raw)", "🥈 Strieborná vrstva (Sentiment)", "🥇 Zlatá vrstva (Aspekty)"])

    # 1. RAW DÁTA
    with tab1:
        st.markdown("### `raw_reviews`")
        st.markdown("Všetky stiahnuté surové recenzie z webu.")
        st.dataframe(raw_df, use_container_width=True, height=400)
        
        # Tlačidlo na stiahnutie CSV
        csv1 = raw_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Stiahnuť raw_reviews (.csv)", 
            data=csv1, 
            file_name='raw_reviews.csv', 
            mime='text/csv',
            key='download_raw'
        )

    # 2. SENTIMENT
    with tab2:
        st.markdown("### `processed_sentiment`")
        st.markdown("Výsledok celkového sentimentu pre každú zanalyzovanú recenziu.")
        st.dataframe(sent_df, use_container_width=True, height=400)
        
        csv2 = sent_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Stiahnuť processed_sentiment (.csv)", 
            data=csv2, 
            file_name='processed_sentiment.csv', 
            mime='text/csv',
            key='download_sent'
        )

    # 3. ASPEKTY
    with tab3:
        st.markdown("### `aspect_analysis`")
        st.markdown("Konkrétne vety a ich priradené kľúčové aspekty z Fuzzy Matchingu.")
        st.dataframe(asp_df, use_container_width=True, height=400)
        
        csv3 = asp_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Stiahnuť aspect_analysis (.csv)", 
            data=csv3, 
            file_name='aspect_analysis.csv', 
            mime='text/csv',
            key='download_asp'
        )

if __name__ == "__main__":
    main()