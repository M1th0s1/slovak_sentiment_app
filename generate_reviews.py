import sqlite3
import random
import uuid
from datetime import datetime, timedelta

def generate_100_reviews():
    conn = sqlite3.connect('decathlon_warehouse.db')
    c = conn.cursor()

    pobocky = ["Bratislava - Eurovea", "Bratislava - Bory", "Zilina", "Kosice", "Nitra", 
                "Banska Bystrica", "Trnava", "Presov", "Trencin"]
    
    # Rozšírené šablóny pre lepšiu diverzitu dát
    templates = [
        ("Bicykel je super, ale obsluha bola hrozne pomalá a neochotná.", 2, "Personál"),
        ("Veľmi milý personál, poradili mi s výberom tenisiek na behanie.", 5, "Personál"),
        ("Hrozný neporiadok na predajni, nevedel som nájsť správnu veľkosť trička.", 1, "Predajňa"),
        ("Ceny sú bezkonkurenčné, Decathlon nesklamal.", 5, "Cena"),
        ("E-shop doručenie trvalo týždeň, balík prišiel poškodený.", 2, "E-shop"),
        ("Skvelý výber stanov a kempingového vybavenia na jednom mieste.", 4, "Produkty"),
        ("Predavač v sekcii cyklistiky bol drzý, sem sa už nevrátim.", 1, "Personál"),
        ("Pomer cena a výkon je pri značke Quechua fakt top.", 5, "Cena"),
        ("Kabínky na skúšanie boli špinavé a bolo ich málo.", 2, "Predajňa"),
        ("Rýchly nákup, samoobslužné pokladne fungujú skvele.", 5, "Predajňa"),
        ("Produkt prišiel nefunkčný, ale reklamácia na pobočke bola blesková.", 4, "Produkty"),
        ("Hľadala som detské lyžiarky, výber bol dosť slabý.", 3, "Produkty")
    ]

    print("🚀 Štartujem generovanie 100 recenzií...")

    start_date = datetime(2025, 1, 1)
    
    for i in range(100):  # ZMENENÉ TU
        rid = str(uuid.uuid4())[:12]
        pob = random.choice(pobocky)
        aut = random.choice(["Marek", "Zuzana", "Lucia", "Peter", "Michal", "Jana", "Igor", "Monika"]) + f" {random.randint(1, 999)}"
        
        tpl, stars, _ = random.choice(templates)
        
        # Pridanie náhodného "vibe-u" na koniec
        vibe = random.choice([" Určite odporúčam!", " Som sklamaný.", " Palec hore.", " Nikdy viac.", " Priemerný zážitok.", " Spokojnosť.", " Otrasné.", ""] )
        text = tpl + vibe
        
        # Náhodný dátum v poslednom roku
        random_days = random.randint(0, 400)
        curr_date = (start_date + timedelta(days=random_days)).strftime("%Y-%m-%d")

        try:
            c.execute("INSERT INTO raw_reviews VALUES (?,?,?,?,?,?)", (rid, pob, aut, text, stars, curr_date))
        except sqlite3.IntegrityError:
            pass

    conn.commit()
    conn.close()
    print("✅ Hotovo! 100 recenzií naliatych do Bronzovej vrstvy (raw_reviews).")

if __name__ == "__main__":
    generate_100_reviews()  # ZMENENÉ TU