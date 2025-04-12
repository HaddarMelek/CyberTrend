import pandas as pd
import os
import re
import pycountry
file_path = "/Users/macbook/ML/pfa/CybeTrend/cyber_data.csv"

if not os.path.exists(file_path):
    print(f"ERROR: The file {file_path} does not exist. Please check the path!")
else:
    print("File found, loading...")
    df = pd.read_csv(file_path)
    print("Preview of the first rows:")
    print(df.head())


def clean_country_name(name):
    patterns = [
        (r"^Independent State of ", ""),
        (r"^Arab Republic of ", ""),
        (r"^Islamic Republic of ", ""),
        (r"^Democratic Republic of ", ""),
        (r"^Republic of ", ""),
        (r"^Kingdom of ", ""),
        (r"^State of ", ""),
        (r"^Plurinational State of ", ""),
        (r"^Socialist Republic of ", ""),
        (r"^United Mexican States", "Mexico"),
        (r"^United States of America", "USA"),
        (r"^United Kingdom of Great Britain and Northern Ireland", "UK"),
        (r"Argentine Republic", "Argentina"),
        (r"Bailiwick of Guernsey", "Guernsey"),
        (r"Bailiwick of Jersey", "Jersey"),
        (r"Bolivarian Republic of Venezuela", "Venezuela"),
        (r"Collectivity of Saint Martin", "Saint Martin"),
        (r"Commonwealth of Puerto Rico", "Puerto Rico"),
        (r"Commonwealth of the Bahamas", "Bahamas"),
        (r"Congo \(Democratic Republic of the\)", "Congo"),
        (r"Co-operative Republic of Guyana", "Guyana"),
        (r"Country of Curaçao", "Curaçao"),
        (r"Czech Republic", "Czech"),
        (r"Democratic Socialist Republic of Sri Lanka", "Sri Lanka"),
        (r"Department of Mayotte", "Mayotte"),
        (r"Dominican Republic", "Dominican"),
        (r"Federal Democratic Republic of Ethiopia", "Ethiopia"),
        (r"Federal Democratic Republic of Nepal", "Nepal"),
        (r"Federal Republic of Germany", "Germany"),
        (r"Federal Republic of Nigeria", "Nigeria"),
        (r"Federal Republic of Somalia", "Somalia"),
        (r"Federative Republic of Brazil", "Brazil"),
        (r"French Republic", "French"),
        (r"Gabonese Republic", "Gabon"),
        (r"Grand Duchy of Luxembourg", "Luxembourg"),
        (r"Hashemite Kingdom of Jordan", "Jordan"),
        (r"Hong Kong Special Administrative Region of the People's Republic of China", "Hong Kong"),
        (r"the Netherlands", "Netherlands"),
        (r"Kyrgyz Republic", "Kyrgyz"),
        (r"Lao People's Democratic Republic", "Lao"),
        (r"Lebanese Republic", "Lebanese"),
        (r"Macao Special Administrative Region of the People's Republic of China", "Macao"),
        (r"Most Serene Republic of San Marino", "San Marino"),
        (r"Nation of Brunei, Abode of Peace", "Brunei"),
        (r"Oriental Republic of Uruguay", "Uruguay"),
        (r"People's Democratic Republic of Algeria", "Algeria"),
        (r"People's Republic of Bangladesh", "Bangladesh"),
        (r"People's Republic of China", "China"),
        (r"Principality of Andorra", "Andorra"),
        (r"Principality of Liechtenstein", "Liechtenstein"),
        (r"Principality of Monaco", "Monaco"),
        (r"China \(Taiwan\)", "Taiwan"),
        (r"the Congo", "Congo"),
        (r"the Gambia", "Gambia"),
        (r"the Maldives", "Maldives"),
        (r"the Philippines", "Philippines"),
        (r"the Sudan", "Sudan"),
        (r"the Union of Myanmar", "Myanmar"),
        (r"Sultanate of Oman", "Oman"),
        (r"Swiss Confederation", "Swiss"),
        (r"Syrian Arab Republic", "Syrian"),
        (r"Territory of Norfolk Island", "Norfolk Island"),
        (r"United Republic of Tanzania", "Tanzania"),
        (r"Commonwealth of Dominica", "Dominica"),
        (r"Union of the Comoros", "Comoros"),
        (r"Commonwealth of the Northern Mariana Islands", "Northern Mariana Islands"),
        (r"Virgin Islands of the United States", "Virgin Islands"),
        (r"Federation of Saint Christopher and Nevisa", "Saint Christopher and Nevisa"),
        (r"Independent and Sovereign Republic of Kiribati", "Kiribati"),
        (r"Federated States of Micronesia", "Micronesia"),
        (r"Vatican City State", "Vatican City"),
        (r"Democratic People's Republic of Korea", "Korea"),
    ]
    for pattern, replacement in patterns:
        name = re.sub(pattern, replacement, name)
    return name.strip()

def get_country_code(country_name):
    try:
        return pycountry.countries.lookup(country_name).alpha_2
    except LookupError:
        return None

df["cleaned_country"] = df["country"].apply(clean_country_name)
df["country_code"] = df["cleaned_country"].apply(get_country_code)

print(df[["country", "cleaned_country", "country_code"]].head())
