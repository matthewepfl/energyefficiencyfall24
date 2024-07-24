import sklearn as sk
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import re
#from text.text_helper import * 

# Set the seed
np.random.seed(42)

def merge_numbappt(listings_df, df_leerstand):
    df_leerstand = df_leerstand[df_leerstand["Anzahl Wohnräume"]
                                    == "Anzahl Wohnräume - Total"]
    df_leerstand = df_leerstand[df_leerstand["Leerwohnung (Typ)"]
                                    == "Leer stehende Wohnung - Total"]
    df_leerstand = df_leerstand[df_leerstand["Anzahl/Anteil"]
                                    == "Anzahl"]
    df_leerstand["Jahr"] = df_leerstand["Jahr"].apply(pd.to_numeric)
    df_leerstand = df_leerstand[df_leerstand["Jahr"] >= 2015]
    # Add column "Number of apartments"
    df_leerstand = df_leerstand.drop(
        columns=["Anzahl Wohnräume", "Leerwohnung (Typ)", "Anzahl/Anteil"])
    df_leerstand = df_leerstand.rename(columns={
                                        "Grossregion (<<) / Kanton (-) / Gemeinde (......)": "Geo City", "DATA": "Number of apartments"})

    # Remove parentheses around Kanton abbreviations
    df_leerstand["Geo City"] = df_leerstand["Geo City"].apply(lambda x: re.sub(
        r"^......[0-9]{4} (\w*) \(([A-Z]{2})\)$", r"\1 \2", x))
    # Remove ......1234 at the beginning
    df_leerstand["Geo City"] = df_leerstand["Geo City"].apply(
        lambda x: re.sub(r"^......[0-9]{4} (.*)$", r"\1", x))

    # Listing data

    listings_df["Jahr"] = listings_df["Day of Advertisement Created"].apply(
        lambda x: pd.to_numeric(x[-4:]))

    municipalities = {"Emmenbrücke": "Emmen", "Pfäffikon SZ": "Freienbach", "Jona": "Rapperswil-Jona", "Glattbrugg": "Opfikon", "Rapperswil SG": "Rapperswil-Jona", "Schliern b. Köniz": "Köniz", "Goldau": "Arth", "Nussbaumen AG": "Obersiggenthal", "Au ZH": "Wädenswil", "Gattikon": "Thalwil", "Binz": "Maur", "Les Acacias": "Genève", "Petit-Lancy": "Lancy", "Lüchingen": "Altstätten", "Viganello": "Lugano", "Anglikon": "Wohlen AG", "Châtelaine": "Vernier", "Pfaffhausen": "Fällanden", "Egg b. Zürich": "Egg", "Brunnen": "Ingenbohl", "Wohlen": "Wohlen AG", "Colombier NE": "Milvignes", "Giubiasco": "Bellinzona", "Altstätten SG": "Altstätten", "S. Antonino": "Sant'Antonino", "Bad Zurzach": "Zurzach", "Brugg AG": "Brugg", "Muri b. Bern": "Muri bei Bern", "Rudolfstetten": "Rudolfstetten-Friedlisberg", "Esslingen": "Egg", "Zollikerberg": "Zollikon", "Grand-Lancy": "Lancy", "Pfäffikon ZH": "Pfäffikon", "Aarau Rohr": "Aarau", "Gunzwil": "Beromünster", "Territet": "Montreux", "Niederglatt ZH": "Niederglatt", "Châtel-St-Denis": "Châtel-Saint-Denis", "Glattpark (Opfikon)": "Opfikon", "Oberkirch LU": "Oberkirch", "Feldbrunnen": "Feldbrunnen-St. Niklaus", "Clarens": "Montreux", "Mollis": "Glarus Nord", "Mont-sur-Lausanne": "Le Mont-sur-Lausanne", "Rombach": "Küttigen", "Oberglatt ZH": "Oberglatt", "Breganzona": "Lugano", "Fahrweid": "Weiningen ZH", "Nussbaumen b. Baden": "Obersiggenthal", "Uitikon Waldegg": "Uitikon", "Dänikon ZH": "Dänikon", "Arbedo": "Arbedo-Castione", "Liebefeld": "Köniz", "Pregassona": "Lugano", "Lustmühle": "Teufen AR", "Effretikon": "Illnau-Effretikon", "Gümligen": "Muri bei Bern", "Castione": "Arbedo-Castione"}

    listings_df["Geo City"] = listings_df["Geo City"].replace(municipalities)

    listings_df = listings_df.merge(right=df_leerstand, on=[
                    "Geo City", "Jahr"], how="left")
    return listings_df

def split_data(data, test_size=0.2, price_pred=False):
    """Split data into training and test sets."""
    # Split the data into training and test sets. (0.2 means 20% of the data is used for testing)
    train_data, test_data = train_test_split(data, test_size=test_size, random_state=42)
    if price_pred:
        # Separate the features and targets
        X_train = train_data.drop(['Price Gross Normalized'], axis=1)
        y_train = train_data['Price Gross Normalized']
        X_test = test_data.drop(['Price Gross Normalized'], axis=1)
        y_test = test_data['Price Gross Normalized']
    else:
        y_train = train_data['Demand']
        X_train = train_data.drop(['Demand'], axis=1)
        y_test  = test_data['Demand']
        X_test  = test_data.drop(['Demand'], axis=1)
    # Print training and test data shapes
    print("Number of training samples: ", X_train.shape[0])
    print("Number of test samples: ", X_test.shape[0])
    return X_train, y_train, X_test, y_test

