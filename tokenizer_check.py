from transformers import AutoTokenizer
import nltk
from nltk.tokenize import word_tokenize

# Uisti sa, že máme stiahnuté dáta pre NLTK
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

# Naša testovacia veta
text = "Najfantastickejší nepremokavý ruksak!"

# 1. Klasický NLTK Tokenizér (Hľadá medzery a interpunkciu)
nltk_tokens = word_tokenize(text, language='slovene')

# 2. Moderný Hugging Face Tokenizér (Algoritmus SentencePiece pre XLM-RoBERTa)
model_path = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
hf_tokenizer = AutoTokenizer.from_pretrained(model_path)
hf_tokens = hf_tokenizer.tokenize(text)

# Výpis výsledkov
print(f"Pôvodný text: {text}\n")

print("--- 1. NLTK Tokenizér (Word-level) ---")
print("Výsledok:", nltk_tokens)
print("Počet tokenov:", len(nltk_tokens))

print("\n--- 2. Hugging Face Tokenizér (Subword-level) ---")
print("Výsledok:", hf_tokens)
print("Počet tokenov:", len(hf_tokens))