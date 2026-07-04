"""
Generate a synthetic, multilingual dataset of short 'place value' statements
labelled with one of three value types (intrinsic / instrumental / relational).

This is SYNTHETIC data, hand-built from templates purely to demonstrate the
classification + validation pipeline. It contains no real survey responses.

The three value types follow the IPBES framing:
  - intrinsic   : nature has worth in itself, independent of human use
  - instrumental: nature as a means to a human end (a benefit/resource)
  - relational  : the meaningfulness of the human-place relationship itself
"""

import csv
import random
from pathlib import Path

random.seed(42)

# ---- Templates per value type, in three languages (nl / de / en) ----
# Each template is a short statement someone might write about a valued place.

TEMPLATES = {
    "intrinsic": {
        "nl": [
            "De natuur hier heeft waarde op zichzelf, los van wat wij eraan hebben.",
            "Dit gebied zou beschermd moeten worden, ook als niemand er ooit komt.",
            "De zeehonden en vogels hebben hun eigen recht om hier te bestaan.",
            "Het landschap is waardevol in zichzelf, niet omdat het ons iets oplevert.",
            "Deze soorten verdienen bescherming ongeacht hun nut voor mensen.",
        ],
        "de": [
            "Die Natur hier hat einen Wert an sich, unabhaengig vom Menschen.",
            "Dieses Gebiet sollte geschuetzt werden, auch wenn niemand hierher kommt.",
            "Die Tiere haben ein eigenes Recht, hier zu leben.",
            "Die Landschaft ist an sich wertvoll, nicht wegen ihres Nutzens.",
            "Diese Arten verdienen Schutz, unabhaengig von ihrem Nutzen fuer uns.",
        ],
        "en": [
            "Nature here has value in itself, regardless of what we get from it.",
            "This area should be protected even if no one ever visits it.",
            "The seals and birds have their own right to exist here.",
            "The landscape is worth protecting for its own sake, not for our benefit.",
            "These species deserve protection whatever their use to people.",
        ],
    },
    "instrumental": {
        "nl": [
            "Dit bos is belangrijk omdat het CO2 opslaat voor ons klimaat.",
            "Ik waardeer dit gebied omdat het schoon drinkwater levert.",
            "De duinen beschermen ons dorp tegen overstromingen.",
            "Deze plek is handig voor recreatie en trekt toeristen aan.",
            "Het wad levert vis en dat is goed voor de lokale economie.",
        ],
        "de": [
            "Dieser Wald ist wichtig, weil er CO2 fuer unser Klima speichert.",
            "Ich schaetze dieses Gebiet, weil es sauberes Trinkwasser liefert.",
            "Die Duenen schuetzen unser Dorf vor Ueberschwemmungen.",
            "Dieser Ort ist nuetzlich fuer Erholung und zieht Touristen an.",
            "Das Watt liefert Fisch, das ist gut fuer die lokale Wirtschaft.",
        ],
        "en": [
            "This forest matters because it stores carbon for our climate.",
            "I value this area because it provides clean drinking water.",
            "The dunes protect our village from flooding.",
            "This place is useful for recreation and brings in tourists.",
            "The mudflats supply fish, which is good for the local economy.",
        ],
    },
    "relational": {
        "nl": [
            "Hier wandelde ik als kind met mijn opa, deze plek is deel van wie ik ben.",
            "Elk jaar komen we hier samen, het houdt onze familie verbonden.",
            "Ik voel me verantwoordelijk voor dit gebied, alsof het van ons is.",
            "Deze plek draagt de herinneringen van ons dorp met zich mee.",
            "Voor mij gaat het om de band die ik met dit landschap heb.",
        ],
        "de": [
            "Hier bin ich als Kind mit meinem Opa gewandert, dieser Ort gehoert zu mir.",
            "Jedes Jahr kommen wir hierher, es haelt unsere Familie zusammen.",
            "Ich fuehle mich fuer dieses Gebiet verantwortlich, als waere es unseres.",
            "Dieser Ort traegt die Erinnerungen unseres Dorfes in sich.",
            "Fuer mich geht es um die Verbindung, die ich zu dieser Landschaft habe.",
        ],
        "en": [
            "I walked here with my grandfather as a child; this place is part of who I am.",
            "Every year we gather here; it keeps our family connected.",
            "I feel responsible for this area, as if it belongs to us.",
            "This place carries the memories of our whole village.",
            "For me it is about the bond I have with this landscape.",
        ],
    },
}

LANG_WEIGHTS = {"nl": 0.5, "de": 0.3, "en": 0.2}  # mimic a Dutch-led survey
DISTANCE_LEVELS = [1, 2, 3, 4]  # 1 = nearest, 4 = farthest (as in a PPGIS design)


def make_dataset(n=300):
    rows = []
    value_types = list(TEMPLATES.keys())
    langs = list(LANG_WEIGHTS.keys())
    lang_p = [LANG_WEIGHTS[l] for l in langs]
    for i in range(n):
        vtype = random.choice(value_types)
        lang = random.choices(langs, weights=lang_p, k=1)[0]
        text = random.choice(TEMPLATES[vtype][lang])
        # relational values are made slightly more common at near distances,
        # instrumental slightly more common at far distances (a plausible prior
        # the pipeline can later be checked against — NOT hard-coded truth).
        if vtype == "relational":
            distance = random.choices(DISTANCE_LEVELS, weights=[4, 3, 2, 1])[0]
        elif vtype == "instrumental":
            distance = random.choices(DISTANCE_LEVELS, weights=[1, 2, 3, 4])[0]
        else:
            distance = random.choice(DISTANCE_LEVELS)
        rows.append({
            "id": f"resp_{i:04d}",
            "language": lang,
            "distance_level": distance,
            "text": text,
            "gold_label": vtype,   # the human 'gold standard' label
        })
    return rows


if __name__ == "__main__":
    out = Path(__file__).resolve().parents[1] / "data" / "synthetic_responses.csv"
    rows = make_dataset(300)
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "language", "distance_level", "text", "gold_label"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} synthetic responses to {out}")
