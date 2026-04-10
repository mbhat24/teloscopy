"""Country-wise and state-wise regional dietary profiles.

Maps frontend region names → backend diet_advisor region_ids, and provides
hierarchical country → state lookups for localised meal planning.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StateProfile:
    """Dietary profile for a state / province within a country."""
    name: str
    staple_foods: list[str] = field(default_factory=list)
    proteins: list[str] = field(default_factory=list)
    vegetables: list[str] = field(default_factory=list)
    fruits: list[str] = field(default_factory=list)
    spices: list[str] = field(default_factory=list)
    traditional_dishes: list[str] = field(default_factory=list)
    dietary_notes: str = ""
    telomere_relevant_foods: list[str] = field(default_factory=list)
    common_deficiencies: list[str] = field(default_factory=list)
    cooking_methods: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class CountryProfile:
    """Dietary profile for a country, with optional state-level detail."""
    name: str
    region_id: str  # maps to diet_advisor GEOGRAPHIC_FOOD_DB region_id
    staple_foods: list[str] = field(default_factory=list)
    proteins: list[str] = field(default_factory=list)
    vegetables: list[str] = field(default_factory=list)
    fruits: list[str] = field(default_factory=list)
    spices: list[str] = field(default_factory=list)
    traditional_dishes: list[str] = field(default_factory=list)
    dietary_notes: str = ""
    states: dict[str, StateProfile] = field(default_factory=dict)
    common_deficiencies: list[str] = field(default_factory=list)
    cultural_restrictions: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Frontend region name → backend region_id mapping
# (fixes the mismatch between the select-option labels and diet_advisor IDs)
# ---------------------------------------------------------------------------

FRONTEND_REGION_MAP: dict[str, str] = {
    # Frontend label              → diet_advisor region_id
    "East Asia":                   "east_asia",
    "South Asia":                  "south_asia_north",
    "Southeast Asia":              "southeast_asia",
    "Central Asia":                "central_asia",
    "Western Asia":                "middle_east",
    "Northern Europe":             "northern_europe",
    "Western Europe":              "western_europe",
    "Eastern Europe":              "eastern_europe",
    "Southern Europe":             "mediterranean",
    "North Africa":                "north_africa",
    "Sub-Saharan Africa":          "sub_saharan_africa",
    "North America":               "north_america",
    "Central America":             "central_america",
    "South America":               "south_america",
    "Oceania":                     "pacific_islands",
    # Also accept raw region_ids (pass-through)
    "east_asia":                   "east_asia",
    "south_asia_north":            "south_asia_north",
    "south_asia_south":            "south_asia_south",
    "south_asia_east":             "south_asia_east",
    "south_asia_west":             "south_asia_west",
    "southeast_asia":              "southeast_asia",
    "central_asia":                "central_asia",
    "middle_east":                 "middle_east",
    "mediterranean":               "mediterranean",
    "northern_europe":             "northern_europe",
    "western_europe":              "western_europe",
    "eastern_europe":              "eastern_europe",
    "sub_saharan_africa":          "sub_saharan_africa",
    "west_africa":                 "west_africa",
    "east_africa":                 "east_africa",
    "north_africa":                "north_africa",
    "southern_africa":             "southern_africa",
    "north_america":               "north_america",
    "central_america":             "central_america",
    "south_america":               "south_america",
    "latin_america":               "latin_america",
    "caribbean":                   "caribbean",
    "pacific_islands":             "pacific_islands",
    "australian":                  "australian",
    "nordic":                      "nordic",
    "balkans":                     "balkans",
    "caucasus":                    "caucasus",
    "central_europe":              "central_europe",
    "andean":                      "andean",
    "iberian":                     "iberian",
}

# ---------------------------------------------------------------------------
# Country → more-specific region_id overrides
# (when a country clearly maps to a sub-region of the broad frontend region)
# ---------------------------------------------------------------------------

_COUNTRY_REGION_OVERRIDE: dict[str, str] = {
    # South Asia sub-regions
    "India":                       "south_asia_south",  # default; states refine further
    "Pakistan":                    "south_asia_north",
    "Bangladesh":                  "south_asia_east",
    "Sri Lanka":                   "south_asia_south",
    "Nepal":                       "south_asia_north",
    # Africa sub-regions
    "Nigeria":                     "west_africa",
    "Ghana":                       "west_africa",
    "Senegal":                     "west_africa",
    "Ivory Coast":                 "west_africa",
    "Cameroon":                    "west_africa",
    "Ethiopia":                    "east_africa",
    "Kenya":                       "east_africa",
    "Tanzania":                    "east_africa",
    "Uganda":                      "east_africa",
    "South Africa":                "southern_africa",
    "Mozambique":                  "southern_africa",
    "Egypt":                       "north_africa",
    "Morocco":                     "north_africa",
    "Tunisia":                     "north_africa",
    "Algeria":                     "north_africa",
    "Libya":                       "north_africa",
    # Europe sub-regions
    "Italy":                       "mediterranean",
    "Spain":                       "iberian",
    "Portugal":                    "iberian",
    "Greece":                      "mediterranean",
    "Sweden":                      "nordic",
    "Norway":                      "nordic",
    "Denmark":                     "nordic",
    "Finland":                     "nordic",
    "Iceland":                     "nordic",
    "United Kingdom":              "northern_europe",
    "Ireland":                     "northern_europe",
    "France":                      "western_europe",
    "Germany":                     "central_europe",
    "Austria":                     "central_europe",
    "Switzerland":                 "western_europe",
    "Netherlands":                 "western_europe",
    "Belgium":                     "western_europe",
    "Russia":                      "eastern_europe",
    "Poland":                      "eastern_europe",
    "Ukraine":                     "eastern_europe",
    "Czech Republic":              "central_europe",
    "Hungary":                     "central_europe",
    "Romania":                     "balkans",
    "Bulgaria":                    "balkans",
    # Americas
    "Mexico":                      "central_america",
    "Brazil":                      "south_america",
    "Argentina":                   "south_america",
    "Colombia":                    "south_america",
    "Peru":                        "andean",
    "Chile":                       "south_america",
    # Asia
    "Turkey":                      "middle_east",
    "Iran":                        "middle_east",
    "Saudi Arabia":                "middle_east",
    "Japan":                       "east_asia",
    "China":                       "east_asia",
    "South Korea":                 "east_asia",
    "Thailand":                    "southeast_asia",
    "Vietnam":                     "southeast_asia",
    "Indonesia":                   "southeast_asia",
    "Philippines":                 "southeast_asia",
    "Malaysia":                    "southeast_asia",
    # Oceania
    "Australia":                   "australian",
    "New Zealand":                 "australian",
}

# ---------------------------------------------------------------------------
# State-level region_id overrides for countries with diverse sub-regions
# ---------------------------------------------------------------------------

_STATE_REGION_OVERRIDE: dict[str, dict[str, str]] = {
    "India": {
        # States with dedicated GeographicProfile in diet_advisor
        "Kerala":                  "india_kerala",
        "Tamil Nadu":              "india_tamil_nadu",
        "Karnataka":               "india_karnataka",
        "Andhra Pradesh":          "india_andhra",
        "Telangana":               "india_andhra",
        "Maharashtra":             "india_maharashtra",
        "Gujarat":                 "india_gujarat",
        "Rajasthan":               "india_rajasthan",
        "Punjab":                  "india_punjab",
        "West Bengal":             "india_west_bengal",
        "Uttar Pradesh":           "india_uttar_pradesh",
        "Goa":                     "india_goa",
        "Odisha":                  "india_odisha",
        # States without dedicated profiles fall back to broad sub-regions
        "Haryana":                 "south_asia_north",
        "Himachal Pradesh":        "south_asia_north",
        "Uttarakhand":             "south_asia_north",
        "Delhi":                   "south_asia_north",
        "Madhya Pradesh":          "south_asia_north",
        "Bihar":                   "south_asia_east",
        "Jharkhand":               "south_asia_east",
        "Assam":                   "south_asia_east",
        "Manipur":                 "south_asia_east",
        "Meghalaya":               "south_asia_east",
        "Tripura":                 "south_asia_east",
        "Mizoram":                 "south_asia_east",
        "Nagaland":                "south_asia_east",
        "Arunachal Pradesh":       "south_asia_east",
        "Sikkim":                  "south_asia_east",
        "Chhattisgarh":            "south_asia_east",
    },
    "China": {
        "Sichuan":                 "china_sichuan",
        "Guangdong":               "china_cantonese",
        "Shandong":                "china_shandong",
        "Shanghai":                "china_jiangsu",
        "Zhejiang":                "china_jiangsu",
        "Fujian":                  "china_cantonese",
        "Hunan":                   "china_hunan",
        "Yunnan":                  "china_sichuan",
        "Beijing":                 "china_shandong",
        "Xinjiang":                "central_asia",
    },
    "Japan": {
        "Okinawa":                 "japan_okinawa",
        "Hokkaido":                "east_asia",
        "Tokyo":                   "east_asia",
        "Osaka":                   "east_asia",
        "Kyoto":                   "east_asia",
        "Hiroshima":               "east_asia",
    },
    "United States": {
        # South
        "Alabama":                 "usa_south",
        "Mississippi":             "usa_south",
        "Louisiana":               "usa_south",
        "Georgia":                 "usa_south",
        "South Carolina":          "usa_south",
        "North Carolina":          "usa_south",
        "Tennessee":               "usa_south",
        "Arkansas":                "usa_south",
        "Kentucky":                "usa_south",
        "Virginia":                "usa_south",
        "West Virginia":           "usa_south",
        "Florida":                 "usa_south",
        # Northeast
        "New York":                "usa_northeast",
        "Massachusetts":           "usa_northeast",
        "Connecticut":             "usa_northeast",
        "New Jersey":              "usa_northeast",
        "Pennsylvania":            "usa_northeast",
        "Rhode Island":            "usa_northeast",
        "New Hampshire":           "usa_northeast",
        "Vermont":                 "usa_northeast",
        "Maine":                   "usa_northeast",
        "Delaware":                "usa_northeast",
        "Maryland":                "usa_northeast",
        # Midwest
        "Ohio":                    "usa_midwest",
        "Illinois":                "usa_midwest",
        "Michigan":                "usa_midwest",
        "Wisconsin":               "usa_midwest",
        "Minnesota":               "usa_midwest",
        "Iowa":                    "usa_midwest",
        "Indiana":                 "usa_midwest",
        "Missouri":                "usa_midwest",
        "Nebraska":                "usa_midwest",
        "Kansas":                  "usa_midwest",
        "North Dakota":            "usa_midwest",
        "South Dakota":            "usa_midwest",
        # West / Pacific
        "California":              "usa_west",
        "Oregon":                  "usa_west",
        "Washington":              "usa_west",
        "Hawaii":                  "usa_west",
        "Alaska":                  "usa_west",
        "Colorado":                "usa_west",
        "Nevada":                  "usa_west",
        "Utah":                    "usa_west",
        "Montana":                 "usa_west",
        "Idaho":                   "usa_west",
        "Wyoming":                 "usa_west",
        # Southwest
        "Texas":                   "usa_southwest",
        "Arizona":                 "usa_southwest",
        "New Mexico":              "usa_southwest",
        "Oklahoma":                "usa_southwest",
    },
    "Mexico": {
        "Mexico City":             "mexico_central",
        "Puebla":                  "mexico_central",
        "Oaxaca":                  "mexico_central",
        "Jalisco":                 "mexico_central",
        "Yucatan":                 "central_america",
        "Veracruz":                "central_america",
        "Nuevo Leon":              "central_america",
        "Baja California":         "central_america",
    },
    "Brazil": {
        "Sao Paulo":               "brazil_southeast",
        "Rio de Janeiro":          "brazil_southeast",
        "Minas Gerais":            "brazil_southeast",
        "Bahia":                   "brazil_northeast",
        "Pernambuco":              "brazil_northeast",
        "Amazonas":                "south_america",
        "Rio Grande do Sul":       "south_america",
    },
    "Italy": {
        "Lombardy":                "italy_north",
        "Veneto":                  "italy_north",
        "Emilia-Romagna":          "italy_north",
        "Tuscany":                 "italy_north",
        "Sicily":                  "italy_south",
        "Campania":                "italy_south",
        "Lazio":                   "italy_south",
    },
}


# =========================================================================
# COMPREHENSIVE COUNTRY PROFILES
# =========================================================================

# ---------- SOUTH ASIA ----------

_INDIA = CountryProfile(
    name="India",
    region_id="south_asia_south",
    staple_foods=["rice", "wheat (roti/chapati)", "millets (ragi, jowar, bajra)"],
    proteins=["lentils/dal", "chickpeas", "paneer", "yogurt/curd", "chicken", "fish (coastal)"],
    vegetables=["spinach", "okra", "eggplant", "bitter gourd", "drumstick/moringa", "cauliflower", "potato"],
    fruits=["mango", "banana", "papaya", "guava", "coconut", "jackfruit"],
    spices=["turmeric", "cumin", "coriander", "black pepper", "cardamom", "fenugreek", "mustard seeds"],
    traditional_dishes=["dal-rice", "roti-sabzi", "biryani", "sambar", "idli-dosa", "rajma chawal"],
    dietary_notes="World's highest vegetarian population (~30-40%). Strong Ayurvedic food traditions.",
    common_deficiencies=["iron", "vitamin_d", "vitamin_b12", "zinc", "iodine"],
    cultural_restrictions=["Hindu: beef avoidance", "Muslim: halal, no pork", "Jain: strict vegetarian, no root vegetables"],
    states={
        "Kerala": StateProfile(
            name="Kerala",
            staple_foods=["rice (boiled/red rice)", "tapioca/cassava"],
            proteins=["fish/seafood (primary)", "chicken", "coconut milk", "lentils"],
            vegetables=["drumstick/moringa", "raw banana", "snake gourd", "ash gourd"],
            fruits=["coconut", "jackfruit", "mango", "papaya"],
            spices=["black pepper", "turmeric", "curry leaves", "cardamom", "cinnamon"],
            traditional_dishes=["meen curry (fish curry)", "avial", "thoran", "puttu", "appam"],
            dietary_notes="Highest fish consumption in India. Beef consumed by Christians & Muslims.",
            telomere_relevant_foods=["fish (omega-3)", "moringa (antioxidant)", "turmeric", "coconut (MCTs)"],
            common_deficiencies=["vitamin_d", "iron"],
            cooking_methods=["coconut oil", "steaming", "coconut milk curries"],
        ),
        "Tamil Nadu": StateProfile(
            name="Tamil Nadu",
            staple_foods=["rice (parboiled)", "millets (ragi/finger millet)"],
            proteins=["lentils (sambar)", "chicken", "fish (coastal)", "curd/yogurt"],
            vegetables=["drumstick", "eggplant", "okra", "bitter gourd"],
            fruits=["banana", "mango", "guava", "tamarind"],
            spices=["black pepper", "mustard seeds", "curry leaves", "fenugreek", "tamarind"],
            traditional_dishes=["sambar", "rasam", "idli/dosa", "pongal", "kootu"],
            dietary_notes="Strong fermented food tradition (idli/dosa batter). Ragi is calcium-rich.",
            telomere_relevant_foods=["ragi (calcium, iron)", "sambar (lentils + vegetables)", "fermented foods"],
            common_deficiencies=["iron", "vitamin_b12", "vitamin_d"],
            cooking_methods=["sesame oil", "groundnut oil", "fermentation", "steaming", "tempering (tadka)"],
        ),
        "Karnataka": StateProfile(
            name="Karnataka",
            staple_foods=["rice", "ragi/finger millet", "jowar/sorghum"],
            proteins=["lentils", "curd/buttermilk", "fish (coastal)", "groundnuts"],
            vegetables=["ridge gourd", "beans", "drumstick", "raw banana", "jackfruit"],
            fruits=["jackfruit", "mango", "sapota/chikoo"],
            spices=["curry leaves", "mustard", "fenugreek", "coconut (coastal)"],
            traditional_dishes=["ragi mudde", "bisi bele bath", "jolada rotti", "kosambari"],
            dietary_notes="Ragi culture — exceptional calcium (344mg/100g). Millet diversity.",
            telomere_relevant_foods=["ragi (calcium, iron, fiber)", "sprouted lentils", "buttermilk"],
            common_deficiencies=["iron", "vitamin_d"],
            cooking_methods=["groundnut oil", "coconut oil (coast)", "steaming"],
        ),
        "Andhra Pradesh": StateProfile(
            name="Andhra Pradesh",
            staple_foods=["rice (dominant — highest per-capita rice consumption)"],
            proteins=["chicken", "eggs", "fish (coastal)", "lentils/dal"],
            vegetables=["drumstick", "eggplant", "green chillies", "bitter gourd"],
            fruits=["tamarind", "raw mango", "guava"],
            spices=["red chili (capsaicin)", "turmeric", "tamarind", "curry leaves", "sesame"],
            traditional_dishes=["biryani", "pesarattu (green gram dosa)", "gongura", "pappu", "gutti vankaya"],
            dietary_notes="Spiciest cuisine in India. Very high chili intake (capsaicin benefits).",
            telomere_relevant_foods=["chili peppers (capsaicin)", "gongura (vitamin C)", "pesarattu (mung protein)"],
            common_deficiencies=["iron", "vitamin_d"],
            cooking_methods=["sesame/sunflower oil", "tempering", "heavy spicing"],
        ),
        "Telangana": StateProfile(
            name="Telangana",
            staple_foods=["rice", "jowar"],
            proteins=["chicken", "mutton", "lentils", "eggs"],
            vegetables=["drumstick", "green chillies", "eggplant"],
            fruits=["mango", "guava"],
            spices=["red chili", "turmeric", "tamarind", "garlic"],
            traditional_dishes=["Hyderabadi biryani", "haleem", "sarva pindi"],
            dietary_notes="Hyderabadi Mughal cuisine influence — rich preparations.",
        ),
        "Maharashtra": StateProfile(
            name="Maharashtra",
            staple_foods=["jowar/sorghum (bhakri)", "rice", "wheat (urban)"],
            proteins=["lentils (amti)", "groundnuts", "fish (Konkan coast)", "sprouted legumes (usal)"],
            vegetables=["drumstick", "eggplant", "kokum (Garcinia indica)"],
            fruits=["raw mango", "Alphonso mango", "banana"],
            spices=["kokum", "goda masala", "turmeric", "asafoetida", "mustard", "curry leaves"],
            traditional_dishes=["varan-bhaat", "misal pav", "poha", "sol kadhi", "pithla"],
            dietary_notes="Jowar bhakri is gluten-free, high-fiber staple. Kokum has anti-obesity properties.",
            telomere_relevant_foods=["jowar (fiber, iron)", "sprouted legumes", "kokum (anti-inflammatory)"],
            common_deficiencies=["iron", "vitamin_d", "vitamin_b12"],
            cooking_methods=["groundnut oil", "shallow frying", "tempering"],
        ),
        "Gujarat": StateProfile(
            name="Gujarat",
            staple_foods=["wheat (rotli)", "bajra/pearl millet", "rice", "jowar"],
            proteins=["lentils/dal", "kadhi (yogurt curry)", "buttermilk", "groundnuts", "chickpea flour"],
            vegetables=["bottle gourd", "ridge gourd", "bitter gourd", "potato", "drumstick"],
            fruits=["dates", "banana", "raw mango"],
            spices=["turmeric", "ajwain/carom seeds", "sesame", "fenugreek", "jaggery"],
            traditional_dishes=["dhokla", "undhiyu", "thepla", "handvo", "kadhi"],
            dietary_notes="~60% vegetarian. Strong fermented food tradition (dhokla, handvo). Sweet-sour-spicy profile.",
            telomere_relevant_foods=["dhokla (fermented)", "buttermilk", "bajra (iron, zinc)", "fenugreek"],
            common_deficiencies=["vitamin_b12", "vitamin_d", "iron"],
            cooking_methods=["groundnut oil", "steaming (dhokla)", "tempering"],
        ),
        "Rajasthan": StateProfile(
            name="Rajasthan",
            staple_foods=["bajra/pearl millet", "wheat", "corn (makki)", "barley"],
            proteins=["lentils", "milk/buttermilk", "ghee", "yogurt", "dried legumes"],
            vegetables=["ker (desert berry)", "sangri (desert bean)", "dried vegetables", "onions"],
            fruits=["ber (Indian jujube)", "amla"],
            spices=["red chili", "garlic", "cumin", "coriander", "dried ginger"],
            traditional_dishes=["dal-baati-churma", "ker sangri", "gatte ki sabzi", "bajra roti"],
            dietary_notes="Arid climate — dried food preservation. High dairy/ghee intake. Bajra is iron-rich.",
            telomere_relevant_foods=["bajra (iron, zinc, fiber)", "amla (vitamin C)", "ghee (fat-soluble vitamins)"],
            common_deficiencies=["vitamin_c", "vitamin_d", "folate"],
            cooking_methods=["ghee", "dry roasting", "baking (baati)", "sun-drying"],
        ),
        "Punjab": StateProfile(
            name="Punjab",
            staple_foods=["wheat (roti/naan)", "rice", "corn (makki — winter)"],
            proteins=["dairy (lassi, paneer, butter, ghee)", "lentils (dal makhani, rajma)", "chicken (tandoori)", "mutton"],
            vegetables=["mustard greens (sarson)", "spinach", "potato", "cauliflower", "peas"],
            fruits=["oranges", "guava"],
            spices=["turmeric", "ginger", "garlic", "coriander", "cumin", "fenugreek"],
            traditional_dishes=["sarson ka saag + makki ki roti", "dal makhani", "rajma chawal", "tandoori chicken", "lassi"],
            dietary_notes="Highest dairy consumption region. Tandoori cooking is healthy. Large portions.",
            telomere_relevant_foods=["sarson ka saag (iron, calcium, folate)", "rajma (plant protein + fiber)", "tandoori cooking"],
            common_deficiencies=["vitamin_d"],
            cooking_methods=["tandoor (clay oven)", "butter/ghee-heavy", "deep frying"],
        ),
        "West Bengal": StateProfile(
            name="West Bengal",
            staple_foods=["rice (dominant — multiple varieties)", "puffed rice (muri)"],
            proteins=["fish (primary — hilsa omega-3 rich)", "lentils (red lentils)", "shrimp", "chicken"],
            vegetables=["bitter gourd", "pointed gourd", "raw banana", "spinach", "pumpkin"],
            fruits=["lychee", "mango", "jackfruit"],
            spices=["mustard (oil and seeds)", "panch phoron (five-spice)", "turmeric", "nigella seeds"],
            traditional_dishes=["maach bhaat (fish-rice)", "shukto", "dal", "posto"],
            dietary_notes="Fish is identity food. Mustard oil is primary fat (rich in omega-3 ALA).",
            telomere_relevant_foods=["hilsa fish (omega-3)", "mustard oil (MUFA, omega-3)", "red lentils (folate)"],
            common_deficiencies=["iron", "vitamin_d"],
            cooking_methods=["mustard oil", "steaming", "light frying"],
        ),
        "Uttar Pradesh": StateProfile(
            name="Uttar Pradesh",
            staple_foods=["wheat (roti/chapati)", "rice"],
            proteins=["lentils (arhar, chana dal)", "milk/curd", "chickpeas", "chicken (Awadhi cuisine)"],
            vegetables=["potato (dominant)", "cauliflower", "peas", "spinach", "bitter gourd"],
            fruits=["mango (Dasheri, Chausa)", "guava"],
            spices=["turmeric", "cumin", "coriander", "ginger", "garlic"],
            traditional_dishes=["roti-dal-sabzi", "Lucknowi biryani", "kebabs", "chaat", "sattu"],
            dietary_notes="Sattu (roasted chickpea flour) is an excellent protein supplement.",
            telomere_relevant_foods=["sattu (protein, minerals)", "dal-roti (complementary protein)"],
            common_deficiencies=["iron", "vitamin_d", "vitamin_b12", "zinc"],
            cooking_methods=["mustard oil (eastern UP)", "ghee", "deep frying", "tandoor"],
        ),
        "Odisha": StateProfile(
            name="Odisha",
            staple_foods=["rice", "millets (ragi, kutki)"],
            proteins=["lentils", "freshwater fish", "dried fish"],
            vegetables=["wild greens (foraged)", "drumstick", "bamboo shoots"],
            fruits=["jackfruit", "mango"],
            spices=["turmeric", "mustard", "panch phoron"],
            traditional_dishes=["pakhala (fermented rice water)", "dalma", "leafy green saag"],
            dietary_notes="Pakhala (fermented rice) is an excellent probiotic food.",
            telomere_relevant_foods=["pakhala (probiotic)", "wild foraged foods", "millets"],
            common_deficiencies=["iron", "protein", "vitamin_a"],
        ),
        "Goa": StateProfile(
            name="Goa",
            staple_foods=["rice", "bread (pao — Portuguese influence)"],
            proteins=["fish/seafood (primary)", "pork (vindaloo)", "chicken", "coconut milk"],
            vegetables=["kokum", "drumstick", "bitter gourd"],
            fruits=["coconut", "cashew fruit", "mango", "jackfruit"],
            spices=["kokum", "tamarind", "chili", "cinnamon", "clove"],
            traditional_dishes=["fish curry rice", "vindaloo", "xacuti", "bebinca"],
            dietary_notes="Portuguese-influenced cuisine. High seafood intake.",
            telomere_relevant_foods=["fish (omega-3)", "kokum (anti-inflammatory)", "coconut"],
        ),
        "Himachal Pradesh": StateProfile(
            name="Himachal Pradesh",
            staple_foods=["wheat", "rice", "corn", "barley"],
            proteins=["lentils", "rajma (kidney beans)", "dairy", "chicken", "trout (river fish)"],
            vegetables=["potato", "peas", "cabbage", "spinach"],
            fruits=["apple (primary crop)", "plum", "apricot", "walnut"],
            spices=["turmeric", "coriander", "red chili"],
            traditional_dishes=["siddu (wheat bread)", "dham (festive platter)", "madra (chickpea yogurt curry)"],
            dietary_notes="Apple region — good antioxidant source. River trout for omega-3.",
            telomere_relevant_foods=["apple (quercetin)", "walnuts (omega-3)", "trout (omega-3)"],
        ),
        "Assam": StateProfile(
            name="Assam",
            staple_foods=["rice (multiple varieties)", "bamboo shoots"],
            proteins=["fish (primary)", "pork", "pigeon", "duck", "lentils"],
            vegetables=["leafy greens", "bamboo shoots", "elephant apple", "ash gourd"],
            fruits=["banana", "jackfruit", "citrus"],
            spices=["khar (alkaline)", "bhut jolokia (ghost pepper)", "ginger"],
            traditional_dishes=["masor tenga (fish sour curry)", "khar", "pitha (rice cake)"],
            dietary_notes="Strong fish culture. Bhut jolokia (ghost pepper) has extreme capsaicin.",
        ),
        "Sikkim": StateProfile(
            name="Sikkim",
            staple_foods=["rice", "millet", "buckwheat", "corn"],
            proteins=["fermented soybean (kinema)", "pork", "yak meat", "churpi (cheese)"],
            vegetables=["fern fiddleheads", "bamboo shoots", "nettle", "wild greens"],
            fruits=["orange", "cardamom"],
            spices=["Sichuan pepper (timur)", "ginger", "garlic"],
            traditional_dishes=["gundruk (fermented greens)", "kinema curry", "thukpa"],
            dietary_notes="Tibetan/Nepali influence. Rich fermented food tradition.",
            telomere_relevant_foods=["fermented foods (kinema, gundruk)", "buckwheat (rutin)"],
        ),
    },
)

_PAKISTAN = CountryProfile(
    name="Pakistan",
    region_id="south_asia_north",
    staple_foods=["wheat (roti/naan)", "rice", "corn"],
    proteins=["lentils/dal", "chicken", "mutton/goat", "beef", "yogurt"],
    vegetables=["potato", "spinach", "okra", "bitter gourd", "onion"],
    fruits=["mango", "guava", "pomegranate", "dates", "citrus"],
    spices=["turmeric", "cumin", "coriander", "chili", "garam masala"],
    traditional_dishes=["biryani", "nihari", "haleem", "sajji", "chapli kebab"],
    dietary_notes="Halal dietary requirements. Meat consumption higher than India.",
    common_deficiencies=["iron", "vitamin_d", "vitamin_a", "zinc"],
    cultural_restrictions=["Muslim: halal, no pork, Ramadan fasting"],
)

_BANGLADESH = CountryProfile(
    name="Bangladesh",
    region_id="south_asia_east",
    staple_foods=["rice (dominant)", "wheat"],
    proteins=["fish (primary — hilsa national fish)", "lentils", "chicken", "eggs"],
    vegetables=["bitter gourd", "leafy greens", "pumpkin", "eggplant"],
    fruits=["jackfruit (national fruit)", "mango", "banana", "litchi"],
    spices=["turmeric", "chili", "mustard", "panch phoron"],
    traditional_dishes=["hilsa fish curry", "dal-bhaat", "pitha", "shutki (dried fish)"],
    dietary_notes="Fish is central to diet. Mustard oil is primary cooking fat.",
    common_deficiencies=["iron", "vitamin_a", "zinc", "iodine"],
    cultural_restrictions=["Muslim: halal, no pork"],
)

_SRI_LANKA = CountryProfile(
    name="Sri Lanka",
    region_id="south_asia_south",
    staple_foods=["rice", "coconut (used in everything)"],
    proteins=["fish/seafood", "lentils (dhal)", "chicken", "coconut milk"],
    vegetables=["jackfruit", "drumstick", "bitter gourd", "pumpkin"],
    fruits=["coconut", "banana", "papaya", "mango", "wood apple"],
    spices=["cinnamon (Ceylon — world's best)", "curry leaves", "pandan", "lemongrass", "goraka"],
    traditional_dishes=["rice and curry", "hoppers (appam)", "kottu roti", "lamprais"],
    dietary_notes="Ceylon cinnamon is premium quality — anti-inflammatory, blood sugar regulating.",
    common_deficiencies=["iron", "vitamin_d", "vitamin_b12"],
)

_NEPAL = CountryProfile(
    name="Nepal",
    region_id="south_asia_north",
    staple_foods=["rice", "wheat", "millet", "buckwheat"],
    proteins=["lentils/dal", "chicken", "goat", "buffalo", "dried fish"],
    vegetables=["spinach", "mustard greens", "potato", "radish", "bamboo shoots"],
    fruits=["orange", "apple (highland)", "banana"],
    spices=["turmeric", "cumin", "Sichuan pepper (timur)", "ginger", "fenugreek"],
    traditional_dishes=["dal bhat tarkari", "momo (dumplings)", "sel roti", "gundruk (fermented greens)", "thukpa"],
    dietary_notes="Dal-bhat-tarkari is daily staple. Strong fermented food tradition (gundruk, sinki).",
    common_deficiencies=["iron", "vitamin_a", "iodine", "vitamin_d"],
)

# ---------- EAST ASIA ----------

_CHINA = CountryProfile(
    name="China",
    region_id="east_asia",
    staple_foods=["rice", "wheat (noodles, steamed buns)", "millet"],
    proteins=["pork", "tofu/soy", "fish/seafood", "chicken", "eggs"],
    vegetables=["bok choy", "napa cabbage", "Chinese broccoli", "mushrooms (shiitake, enoki)", "lotus root", "bamboo shoots"],
    fruits=["persimmon", "mandarin", "lychee", "Asian pear", "jujube"],
    spices=["ginger", "garlic", "star anise", "Sichuan peppercorn", "five-spice", "scallion"],
    traditional_dishes=["kung pao chicken", "mapo tofu", "dim sum", "hot pot", "congee", "Peking duck"],
    dietary_notes="Eight Great Cuisines (八大菜系). High soy intake. Green tea tradition.",
    common_deficiencies=["calcium", "vitamin_d", "iron (women)"],
    cultural_restrictions=["Buddhist vegetarianism (monastic)", "Hui Muslims: halal, no pork"],
    states={
        "Sichuan": StateProfile(
            name="Sichuan",
            staple_foods=["rice", "wheat noodles"],
            proteins=["pork", "freshwater fish", "tofu", "chicken", "rabbit"],
            vegetables=["chili peppers", "garlic", "ginger", "fermented black beans"],
            spices=["Sichuan peppercorn (numbing)", "doubanjiang (chili bean paste)", "chili oil"],
            traditional_dishes=["mapo tofu", "kung pao chicken", "dan dan noodles", "hot pot", "twice-cooked pork"],
            dietary_notes="Sichuan peppercorn has anti-inflammatory and analgesic properties.",
            telomere_relevant_foods=["Sichuan peppercorn (anti-inflammatory)", "garlic (allicin)", "fermented bean paste"],
        ),
        "Guangdong": StateProfile(
            name="Guangdong",
            staple_foods=["rice", "rice noodles", "congee"],
            proteins=["seafood (shrimp, fish, crab)", "pork", "chicken", "duck"],
            vegetables=["Chinese broccoli", "water spinach", "bean sprouts"],
            spices=["ginger", "scallion", "oyster sauce", "soy sauce"],
            traditional_dishes=["dim sum", "wonton noodle soup", "steamed fish", "roast duck", "congee", "double-boiled soup"],
            dietary_notes="Healthiest Chinese cuisine — emphasis on steaming, fresh ingredients. Medicinal soups.",
            telomere_relevant_foods=["steamed fish (omega-3)", "medicinal soups (goji, astragalus)", "green tea"],
        ),
        "Yunnan": StateProfile(
            name="Yunnan",
            staple_foods=["rice", "rice noodles"],
            proteins=["mushrooms (extremely diverse)", "goat", "pork", "freshwater fish"],
            vegetables=["wild mushrooms", "edible flowers", "herbs"],
            fruits=["Pu-erh tea"],
            spices=["herbs", "chili", "Sichuan pepper"],
            traditional_dishes=["crossing-the-bridge noodles", "mushroom hot pot", "steam pot chicken"],
            dietary_notes="World's most diverse mushroom consumption. Pu-erh tea for gut health.",
            telomere_relevant_foods=["wild mushrooms (beta-glucans)", "Pu-erh tea (fermented — gut microbiome)"],
        ),
        "Shandong": StateProfile(
            name="Shandong",
            staple_foods=["wheat (steamed buns, noodles, dumplings)", "corn", "sweet potato"],
            proteins=["seafood (Yellow Sea)", "pork", "chicken", "sea cucumber"],
            vegetables=["garlic", "scallions", "cabbage"],
            spices=["vinegar", "scallions", "garlic", "soy sauce"],
            traditional_dishes=["sweet and sour carp", "braised sea cucumber", "scallion pancakes"],
            dietary_notes="Garlic capital of China. Sea cucumber for collagen/joint health.",
            telomere_relevant_foods=["garlic (allicin)", "sea cucumber (collagen)", "vinegar (blood sugar)"],
        ),
        "Fujian": StateProfile(
            name="Fujian",
            staple_foods=["rice", "sweet potato", "rice noodles"],
            proteins=["seafood (abundant)", "pork", "mushrooms"],
            spices=["fish sauce", "shrimp paste", "red yeast rice", "five-spice powder"],
            traditional_dishes=["oyster omelette", "Buddha jumps over the wall", "popia (spring rolls)"],
            dietary_notes="Red yeast rice contains natural statin (monacolin K) — cardiovascular benefits.",
            telomere_relevant_foods=["red yeast rice (natural statin)", "seafood (omega-3)", "mushrooms"],
        ),
        "Hunan": StateProfile(
            name="Hunan",
            staple_foods=["rice"],
            proteins=["pork", "freshwater fish", "tofu", "smoked meats"],
            vegetables=["fresh chili peppers", "garlic"],
            spices=["fresh chili (not dried)", "fermented black beans", "shallots"],
            traditional_dishes=["red-braised pork", "steamed fish head with chili", "stir-fried vegetables"],
            dietary_notes="Fresh chili provides vitamin C + capsaicin. Smoked meat tradition.",
            telomere_relevant_foods=["fresh chili (vitamin C, capsaicin)"],
        ),
    },
)

_JAPAN = CountryProfile(
    name="Japan",
    region_id="east_asia",
    staple_foods=["rice", "soba (buckwheat)", "udon"],
    proteins=["fish/seafood", "tofu", "natto (fermented soy)", "miso", "eggs", "seaweed"],
    vegetables=["seaweed/kelp", "daikon", "mushrooms (shiitake, maitake)", "edamame"],
    fruits=["persimmon", "mandarin", "yuzu", "plum"],
    spices=["wasabi", "ginger", "shiso", "miso", "soy sauce", "dashi"],
    traditional_dishes=["sushi", "miso soup", "ramen", "tempura", "natto", "kaiseki"],
    dietary_notes="Washoku (traditional Japanese cuisine) is UNESCO heritage. Hara hachi bu (eat 80% full).",
    common_deficiencies=["calcium", "vitamin_d"],
    states={
        "Okinawa": StateProfile(
            name="Okinawa",
            staple_foods=["sweet potato (purple and orange — primary, not rice)", "rice", "soba"],
            proteins=["tofu", "pork (minimal)", "fish", "seaweed", "soy"],
            vegetables=["goya/bitter melon", "seaweed"],
            fruits=["shikuwasa (citrus)"],
            spices=["turmeric", "ginger"],
            traditional_dishes=["champuru (stir-fry)", "goya champuru", "soki soba", "taco rice"],
            dietary_notes="BLUE ZONE — world's highest longevity. Caloric restriction + nutrient density. Purple sweet potato (anthocyanins).",
            telomere_relevant_foods=["purple sweet potato (anthocyanins)", "bitter melon (glucose regulation)", "turmeric (anti-inflammatory)", "seaweed (iodine)"],
            common_deficiencies=[],
        ),
        "Hokkaido": StateProfile(
            name="Hokkaido",
            staple_foods=["rice", "wheat", "potatoes", "corn"],
            proteins=["salmon (wild)", "crab", "sea urchin", "scallop", "dairy", "lamb"],
            vegetables=["corn", "potato", "root vegetables"],
            fruits=["melon", "lavender"],
            traditional_dishes=["miso ramen", "Genghis Khan barbecue (lamb)", "seafood donburi"],
            dietary_notes="Japan's dairy region. Wild Pacific salmon (astaxanthin + omega-3).",
            telomere_relevant_foods=["wild salmon (astaxanthin + omega-3)", "dairy (calcium)"],
        ),
        "Tokyo": StateProfile(
            name="Tokyo",
            staple_foods=["rice", "soba (buckwheat noodles)"],
            proteins=["fish/sushi (Edo-mae tradition)", "natto", "tofu", "seafood"],
            spices=["soy sauce", "wasabi", "mirin"],
            traditional_dishes=["Edo-mae sushi", "soba", "monjayaki", "tempura"],
            dietary_notes="Natto (vitamin K2, nattokinase — cardiovascular/bone health). Soba (rutin).",
            telomere_relevant_foods=["natto (K2, nattokinase)", "sushi (fish omega-3)", "buckwheat soba (rutin)"],
        ),
        "Kyoto": StateProfile(
            name="Kyoto",
            staple_foods=["rice", "udon"],
            proteins=["tofu (Kyoto is tofu capital)", "yuba (tofu skin)", "fish"],
            vegetables=["kyo-yasai (heritage vegetables)", "bamboo shoots"],
            spices=["matcha (Uji — highest grade)", "miso", "dashi (kelp-bonito)"],
            traditional_dishes=["kaiseki", "shojin ryori (temple cuisine)", "yudofu", "matcha sweets"],
            dietary_notes="Matcha (10x antioxidants of green tea). Shojin ryori is entirely plant-based Buddhist cuisine.",
            telomere_relevant_foods=["matcha (EGCG, L-theanine)", "tofu (isoflavones)", "heritage vegetables"],
        ),
    },
)

_SOUTH_KOREA = CountryProfile(
    name="South Korea",
    region_id="east_asia",
    staple_foods=["rice", "barley", "sweet potato"],
    proteins=["tofu", "fish/seafood", "pork (samgyeopsal)", "beef (bulgogi)", "eggs", "seaweed"],
    vegetables=["kimchi (fermented cabbage)", "bean sprouts", "spinach", "zucchini", "perilla leaves"],
    fruits=["Korean pear", "persimmon", "mandarin"],
    spices=["gochugaru (red pepper flakes)", "doenjang (fermented soybean paste)", "gochujang", "sesame", "garlic", "ginger"],
    traditional_dishes=["kimchi", "bibimbap", "bulgogi", "samgyeopsal", "japchae", "doenjang jjigae"],
    dietary_notes="Kimchi in every meal — excellent probiotic. Fermented food culture (doenjang, gochujang).",
    common_deficiencies=["vitamin_d", "calcium"],
)

# ---------- SOUTHEAST ASIA ----------

_THAILAND = CountryProfile(
    name="Thailand",
    region_id="southeast_asia",
    staple_foods=["jasmine rice", "sticky/glutinous rice", "rice noodles"],
    proteins=["fish", "shrimp", "chicken", "pork", "tofu", "eggs"],
    vegetables=["water spinach (kangkung)", "bean sprouts", "Thai basil", "bamboo shoots", "green papaya"],
    fruits=["mango", "durian", "mangosteen", "rambutan", "dragon fruit", "pineapple"],
    spices=["lemongrass", "galangal", "kaffir lime", "Thai chili", "fish sauce", "shrimp paste", "Thai basil"],
    traditional_dishes=["pad thai", "tom yum", "green/red curry", "som tum (papaya salad)", "massaman curry"],
    dietary_notes="Fresh herbs dominant (anti-inflammatory, antioxidant). Fish sauce is universal condiment.",
    common_deficiencies=["iron", "vitamin_a", "iodine"],
    cultural_restrictions=["Buddhist: periodic vegetarianism"],
)

_VIETNAM = CountryProfile(
    name="Vietnam",
    region_id="southeast_asia",
    staple_foods=["rice", "rice noodles", "baguette (French influence)"],
    proteins=["fish/seafood", "pork", "chicken", "tofu", "eggs"],
    vegetables=["herbs (mint, cilantro, Vietnamese coriander)", "bean sprouts", "morning glory", "banana flower"],
    fruits=["dragon fruit", "rambutan", "pomelo", "starfruit", "jackfruit"],
    spices=["fish sauce (nuoc mam)", "lemongrass", "star anise", "cinnamon", "chili"],
    traditional_dishes=["pho", "banh mi", "bun cha", "spring rolls", "cao lau"],
    dietary_notes="Emphasis on fresh herbs and raw vegetables. Lighter than other Southeast Asian cuisines.",
    common_deficiencies=["iron", "vitamin_a", "zinc"],
)

_INDONESIA = CountryProfile(
    name="Indonesia",
    region_id="southeast_asia",
    staple_foods=["rice", "cassava", "corn"],
    proteins=["tempeh (fermented soybean — excellent)", "tofu", "fish", "chicken", "eggs"],
    vegetables=["kangkung (water spinach)", "cassava leaves", "papaya leaves", "bean sprouts"],
    fruits=["banana", "mango", "papaya", "rambutan", "salak (snake fruit)"],
    spices=["galangal", "lemongrass", "turmeric", "candlenut", "shrimp paste", "kecap manis"],
    traditional_dishes=["nasi goreng", "satay", "rendang", "gado-gado", "soto"],
    dietary_notes="Tempeh is Indonesian invention — excellent probiotic/protein. World's largest Muslim population.",
    common_deficiencies=["iron", "vitamin_a", "zinc", "iodine"],
    cultural_restrictions=["Muslim majority: halal, no pork", "Hindu minority (Bali): some beef avoidance"],
)

# ---------- WESTERN ASIA / MIDDLE EAST ----------

_TURKEY = CountryProfile(
    name="Turkey",
    region_id="middle_east",
    staple_foods=["wheat (bread, pita, bulgur)", "rice"],
    proteins=["lamb", "chicken", "fish (coastal)", "chickpeas", "lentils", "yogurt", "white cheese"],
    vegetables=["tomatoes", "peppers", "eggplant", "zucchini", "onion"],
    fruits=["figs", "pomegranates", "apricots", "grapes", "olives"],
    spices=["sumac", "cumin", "paprika", "mint", "oregano", "Aleppo pepper"],
    traditional_dishes=["kebab", "dolma", "manti", "lahmacun", "pide", "borek"],
    dietary_notes="Bridge between Mediterranean and Middle Eastern cuisines. Strong yogurt/fermented dairy culture.",
    common_deficiencies=["vitamin_d", "iron"],
    cultural_restrictions=["Muslim: halal, no pork, Ramadan fasting"],
)

_IRAN = CountryProfile(
    name="Iran",
    region_id="middle_east",
    staple_foods=["rice (saffron rice — primary)", "wheat (flatbreads — lavash, sangak, barbari)"],
    proteins=["lamb", "chicken", "fish", "lentils", "beans", "yogurt", "kashk", "walnuts"],
    vegetables=["herbs (fresh herb platter — sabzi khordan)", "eggplant", "tomato", "spinach"],
    fruits=["pomegranate", "sour cherry", "barberries", "dried fruits (figs, dates, apricots)"],
    spices=["saffron", "turmeric", "cinnamon", "rose water", "dried lime (limoo amani)", "sumac"],
    traditional_dishes=["chelo kebab", "ghormeh sabzi (herb stew)", "fesenjan (walnut-pomegranate)", "tahdig (crispy rice)", "ash reshteh"],
    dietary_notes="Saffron is most expensive spice — anti-depressant, antioxidant properties. Strong nut culture.",
    common_deficiencies=["vitamin_d", "iron", "vitamin_b12"],
    cultural_restrictions=["Muslim: halal, no pork, no alcohol, Ramadan fasting"],
)

_SAUDI_ARABIA = CountryProfile(
    name="Saudi Arabia",
    region_id="middle_east",
    staple_foods=["rice", "wheat (flatbread)"],
    proteins=["lamb", "chicken", "camel (traditional)", "fish (coastal)", "lentils", "yogurt"],
    vegetables=["tomato", "cucumber", "onion"],
    fruits=["dates (primary — hundreds of varieties)", "citrus", "figs"],
    spices=["cardamom", "saffron", "black lime", "baharat blend", "cinnamon"],
    traditional_dishes=["kabsa (spiced rice with meat)", "mandi", "jareesh", "sambousek"],
    dietary_notes="Dates are central — excellent minerals, fiber. Arabic coffee with cardamom tradition.",
    common_deficiencies=["vitamin_d", "iron"],
    cultural_restrictions=["Muslim: halal, no pork, no alcohol, Ramadan fasting"],
)

# ---------- EUROPE ----------

_UK = CountryProfile(
    name="United Kingdom",
    region_id="northern_europe",
    staple_foods=["wheat (bread)", "potatoes", "oats"],
    proteins=["beef", "chicken", "pork", "fish (cod, haddock)", "eggs", "dairy", "baked beans"],
    vegetables=["potatoes", "peas", "carrots", "cabbage", "leeks (Wales)"],
    fruits=["apples", "berries (blackberry, gooseberry, strawberry)"],
    spices=["mint", "mustard", "horseradish", "vinegar"],
    traditional_dishes=["fish and chips", "full English breakfast", "shepherd's pie", "roast dinner", "haggis (Scotland)"],
    dietary_notes="Growing plant-based movement. Multicultural cuisine (Indian food is 'national dish').",
    common_deficiencies=["vitamin_d", "iron (women)", "folate"],
)

_FRANCE = CountryProfile(
    name="France",
    region_id="western_europe",
    staple_foods=["wheat (baguette, bread)", "potatoes"],
    proteins=["cheese (enormous variety)", "beef", "pork", "poultry", "fish", "eggs", "lentils (Le Puy)"],
    vegetables=["leeks", "endive", "green beans", "mushrooms", "asparagus"],
    fruits=["grapes", "apples", "pears", "cherries", "berries"],
    spices=["herbes de Provence", "tarragon", "thyme", "bay leaf", "parsley"],
    traditional_dishes=["coq au vin", "cassoulet", "ratatouille", "bouillabaisse", "croque-monsieur", "soufflé"],
    dietary_notes="French Paradox — high saturated fat with low CVD (wine, portion control, meal structure).",
    common_deficiencies=["vitamin_d", "folate"],
)

_GERMANY = CountryProfile(
    name="Germany",
    region_id="central_europe",
    staple_foods=["rye bread", "wheat", "potatoes"],
    proteins=["pork (dominant)", "beef", "sausages (bratwurst)", "dairy", "eggs"],
    vegetables=["sauerkraut", "cabbage", "potatoes", "beets", "asparagus"],
    fruits=["apples", "plums", "cherries"],
    spices=["caraway", "mustard", "juniper", "horseradish", "parsley"],
    traditional_dishes=["bratwurst", "schnitzel", "sauerbraten", "kartoffelsalat", "sauerkraut"],
    dietary_notes="Strong fermented food tradition (sauerkraut — probiotics). Rye bread is high fiber.",
    common_deficiencies=["vitamin_d", "folate"],
)

_ITALY = CountryProfile(
    name="Italy",
    region_id="mediterranean",
    staple_foods=["wheat pasta (durum)", "bread", "polenta (north)", "rice (risotto — north)"],
    proteins=["fish/seafood", "legumes (chickpeas, lentils, fava)", "cheese (Parmigiano, mozzarella, pecorino)", "prosciutto"],
    vegetables=["tomatoes", "peppers", "eggplant", "zucchini", "artichoke", "garlic", "leafy greens"],
    fruits=["olives", "grapes", "citrus", "figs", "almonds", "pistachios (Sicily)"],
    spices=["basil", "oregano", "rosemary", "sage", "garlic", "red pepper flakes"],
    traditional_dishes=["pasta", "pizza", "risotto", "minestrone", "osso buco", "tiramisu"],
    dietary_notes="Mediterranean diet origin. Southern Italy & Sardinia (Blue Zone) have lowest CVD in Italy.",
    common_deficiencies=["vitamin_d"],
    states={
        "Sicily": StateProfile(
            name="Sicily",
            staple_foods=["wheat pasta", "bread", "couscous (Arab influence)"],
            proteins=["fish/seafood (sardines, swordfish)", "chickpeas", "sheep cheese"],
            vegetables=["eggplant (caponata)", "tomato", "artichoke"],
            fruits=["citrus (blood orange)", "almonds", "pistachios (Bronte)"],
            spices=["saffron", "oregano", "basil"],
            traditional_dishes=["pasta alla norma", "arancini", "caponata", "cassata"],
            dietary_notes="Arab, Greek, Norman influences. Pistachio capital.",
            telomere_relevant_foods=["sardines (omega-3)", "pistachios", "blood oranges (anthocyanins)"],
        ),
        "Tuscany": StateProfile(
            name="Tuscany",
            staple_foods=["bread (unsalted)", "farro", "ribollita"],
            proteins=["white beans (cannellini)", "wild boar", "beef (bistecca)", "pecorino"],
            vegetables=["kale (cavolo nero)", "tomato", "zucchini"],
            fruits=["olive oil (premium)", "grapes (Chianti)"],
            spices=["rosemary", "sage", "garlic"],
            traditional_dishes=["ribollita (bread soup)", "bistecca alla fiorentina", "pappardelle al cinghiale"],
            dietary_notes="Simple, ingredient-focused cooking. Excellent olive oil. Farro (ancient grain).",
            telomere_relevant_foods=["olive oil (MUFA, polyphenols)", "cannellini beans (fiber, protein)", "cavolo nero (antioxidants)"],
        ),
    },
)

_SPAIN = CountryProfile(
    name="Spain",
    region_id="iberian",
    staple_foods=["wheat (bread)", "rice (paella)", "potatoes"],
    proteins=["fish/seafood", "pork (ibérico)", "legumes (chickpeas, lentils)", "eggs", "cheese (manchego)"],
    vegetables=["tomatoes", "peppers", "garlic", "onion", "olive"],
    fruits=["oranges", "grapes", "figs", "almonds", "olives"],
    spices=["saffron", "paprika (pimentón)", "garlic", "bay leaf", "oregano"],
    traditional_dishes=["paella", "gazpacho", "tapas", "tortilla española", "fabada asturiana", "jamón ibérico"],
    dietary_notes="Mediterranean diet. Olive oil is primary fat. Late meal times (lunch 2-3pm, dinner 9-10pm).",
    common_deficiencies=["vitamin_d"],
)

_GREECE = CountryProfile(
    name="Greece",
    region_id="mediterranean",
    staple_foods=["wheat (bread)", "barley"],
    proteins=["fish/seafood", "feta cheese", "yogurt", "lamb", "legumes (fava, chickpeas)"],
    vegetables=["tomatoes", "cucumbers", "olives", "eggplant", "peppers", "wild greens (horta)"],
    fruits=["grapes", "figs", "pomegranates", "citrus", "olives"],
    spices=["oregano", "thyme", "dill", "mint", "garlic", "lemon"],
    traditional_dishes=["moussaka", "souvlaki", "Greek salad", "spanakopita", "dolmades", "baklava"],
    dietary_notes="Classic Mediterranean diet. Wild greens (horta) tradition — exceptional antioxidants.",
    common_deficiencies=["vitamin_d"],
)

# ---------- AFRICA ----------

_NIGERIA = CountryProfile(
    name="Nigeria",
    region_id="west_africa",
    staple_foods=["cassava (garri, fufu)", "yam (pounded yam)", "rice", "plantain", "millet (north)", "sorghum (north)"],
    proteins=["fish (smoked, dried)", "cowpeas/black-eyed peas", "egusi (melon seeds)", "chicken", "goat", "groundnuts (north)"],
    vegetables=["leafy greens (efo/amaranth)", "okra", "tomato", "onion", "pepper"],
    fruits=["mango", "banana", "plantain", "pawpaw/papaya", "orange"],
    spices=["locust beans (dawadawa — fermented)", "scotch bonnet pepper", "palm oil", "crayfish (dried)"],
    traditional_dishes=["jollof rice", "egusi soup", "moin-moin (bean pudding)", "suya (grilled meat)", "pounded yam"],
    dietary_notes="Highly diverse by region and ethnicity. Palm oil provides tocotrienols. Locust beans are probiotic.",
    common_deficiencies=["iron", "vitamin_a", "zinc", "iodine"],
    states={
        "Lagos": StateProfile(
            name="Lagos",
            staple_foods=["rice", "cassava (garri/eba)", "plantain"],
            proteins=["fish", "chicken", "beans", "egusi"],
            vegetables=["efo (spinach/amaranth)", "okra", "tomato"],
            traditional_dishes=["jollof rice", "efo riro", "pepper soup", "suya", "amala"],
            dietary_notes="Urban diet — diverse. Yoruba food culture with diverse leafy soups.",
        ),
        "Kano": StateProfile(
            name="Kano",
            staple_foods=["millet", "sorghum", "rice"],
            proteins=["groundnuts", "cowpeas", "beef", "dairy (nono/fermented milk)"],
            vegetables=["baobab leaves", "moringa"],
            fruits=["dates", "baobab fruit"],
            traditional_dishes=["tuwo (millet porridge)", "miyan kuka (baobab soup)", "kilishi (dried meat)", "fura da nono"],
            dietary_notes="Hausa-Fulani cuisine. Baobab has 6x vitamin C of oranges. Millet is iron/zinc-rich.",
            telomere_relevant_foods=["baobab (vitamin C, calcium)", "millet (iron, zinc, fiber)", "fermented dairy (probiotic)", "moringa"],
        ),
        "Enugu": StateProfile(
            name="Enugu",
            staple_foods=["cassava (fufu, garri)", "yam", "rice"],
            proteins=["fish (stockfish, dried)", "ogiri (fermented sesame)", "beans", "goat"],
            vegetables=["bitter leaf", "oha leaf", "okra"],
            traditional_dishes=["oha soup", "bitter leaf soup", "egusi soup", "abacha"],
            dietary_notes="Bitter leaf (Vernonia amygdalina) has anti-diabetic, hepatoprotective properties.",
            telomere_relevant_foods=["bitter leaf (antioxidant, anti-diabetic)", "fermented condiments"],
        ),
    },
)

_ETHIOPIA = CountryProfile(
    name="Ethiopia",
    region_id="east_africa",
    staple_foods=["teff (injera — uniquely Ethiopian grain)", "sorghum", "barley", "wheat"],
    proteins=["lentils", "chickpeas", "split peas", "beef", "chicken", "eggs"],
    vegetables=["collard greens (gomen)", "cabbage", "carrots", "potatoes"],
    fruits=["banana", "papaya", "mango", "avocado"],
    spices=["berbere (spice blend — chili, fenugreek, coriander)", "mitmita", "niter kibbeh (spiced butter)"],
    traditional_dishes=["injera with wot", "doro wot (chicken stew)", "kitfo (raw beef)", "shiro (chickpea stew)"],
    dietary_notes="Teff is exceptionally high in iron and calcium. Ethiopian Orthodox: ~180 vegan fasting days/year.",
    common_deficiencies=["iron", "vitamin_a", "zinc", "iodine"],
    cultural_restrictions=["Ethiopian Orthodox: extensive vegan fasting (~180 days/year)", "Muslim: halal, no pork"],
)

_KENYA = CountryProfile(
    name="Kenya",
    region_id="east_africa",
    staple_foods=["maize (ugali)", "rice", "wheat (chapati)"],
    proteins=["beans (primary)", "lentils", "beef", "goat", "chicken", "fish (Lake Victoria, coastal)"],
    vegetables=["kale/sukuma wiki", "cabbage", "tomato", "onion"],
    fruits=["mango", "banana", "passion fruit", "avocado"],
    spices=["coriander", "cumin", "chili", "curry powder (coastal — Indian influence)"],
    traditional_dishes=["ugali-sukuma", "nyama choma (grilled meat)", "githeri (corn-bean mix)", "chapati"],
    dietary_notes="Ugali-sukuma wiki is daily staple. Sukuma wiki (kale) provides iron, calcium, vitamin K.",
    common_deficiencies=["iron", "vitamin_a", "zinc"],
)

_SOUTH_AFRICA = CountryProfile(
    name="South Africa",
    region_id="southern_africa",
    staple_foods=["maize (pap/mielie)", "wheat (bread)", "rice"],
    proteins=["beef", "chicken", "lamb", "fish/seafood (coastal)", "beans", "biltong (dried meat)"],
    vegetables=["spinach (morogo)", "pumpkin", "butternut squash", "cabbage"],
    fruits=["citrus", "grapes (wine region)", "subtropical fruits (avocado, mango, litchi)"],
    spices=["curry powder (Cape Malay)", "peri-peri (Portuguese influence)", "coriander"],
    traditional_dishes=["braai (barbecue)", "bobotie (Cape Malay)", "bunny chow", "biltong", "pap and wors"],
    dietary_notes="Rainbow nation — very diverse cuisine. Cape Malay, Dutch, Indian, Indigenous influences.",
    common_deficiencies=["iron", "vitamin_a", "zinc"],
)

# ---------- AMERICAS ----------

_USA = CountryProfile(
    name="United States",
    region_id="north_america",
    staple_foods=["wheat (bread, cereals)", "corn", "rice", "potatoes"],
    proteins=["beef", "chicken", "pork", "dairy", "eggs", "fish (limited inland)", "beans"],
    vegetables=["potatoes", "corn", "lettuce", "tomatoes", "onions"],
    fruits=["apples", "bananas", "oranges", "berries"],
    spices=["salt", "pepper", "garlic powder", "onion powder", "oregano"],
    traditional_dishes=["hamburger", "hot dog", "Thanksgiving turkey", "mac and cheese", "BBQ ribs", "apple pie"],
    dietary_notes="Standard American Diet (SAD) is reference for unhealthy patterns. BUT great diversity available.",
    common_deficiencies=["vitamin_d", "magnesium", "potassium", "fiber"],
    states={
        "California": StateProfile(
            name="California",
            staple_foods=["diverse — health-food culture"],
            proteins=["Pacific salmon (wild)", "tofu/tempeh", "chicken", "avocado"],
            vegetables=["avocado", "kale", "artichoke", "leafy greens", "mushrooms"],
            fruits=["avocado", "citrus", "berries", "grapes (wine)"],
            traditional_dishes=["poke bowls", "avocado toast", "farm-to-table cuisine", "sourdough (SF)", "burritos"],
            dietary_notes="Highest vegetable/fruit consumption in US. Health-conscious food culture.",
            telomere_relevant_foods=["wild salmon (omega-3)", "avocado (MUFA, potassium)", "berries (anthocyanins)", "kale (sulforaphane)"],
        ),
        "Louisiana": StateProfile(
            name="Louisiana",
            staple_foods=["rice", "cornbread"],
            proteins=["crawfish", "shrimp", "catfish", "andouille sausage", "red beans"],
            vegetables=["okra", "bell pepper (trinity)", "celery", "onion"],
            spices=["cayenne", "bay leaf", "thyme", "filé powder"],
            traditional_dishes=["gumbo", "jambalaya", "étouffée", "po'boy", "beignets", "red beans and rice"],
            dietary_notes="Creole/Cajun cuisine — French, African, Spanish influences. Crawfish is lean protein.",
        ),
        "Texas": StateProfile(
            name="Texas",
            staple_foods=["corn tortillas", "wheat tortillas", "rice", "beans"],
            proteins=["beef (ranching)", "pinto/black beans", "chicken", "brisket (BBQ)"],
            vegetables=["chili peppers", "corn", "tomatoes", "cactus (nopal)"],
            spices=["chili powder", "cumin", "garlic", "smoked paprika"],
            traditional_dishes=["BBQ brisket", "Tex-Mex", "chili con carne", "fajitas", "breakfast tacos"],
            dietary_notes="Tex-Mex and BBQ culture. Beans are daily. High red meat consumption.",
            telomere_relevant_foods=["beans (fiber, protein)", "chili peppers (capsaicin)", "avocado"],
        ),
        "Hawaii": StateProfile(
            name="Hawaii",
            staple_foods=["rice", "taro", "sweet potato"],
            proteins=["fish (poke — raw tuna)", "spam (cultural staple)", "tofu", "seaweed"],
            vegetables=["taro", "seaweed", "sweet potato leaves"],
            fruits=["pineapple", "coconut", "passion fruit (lilikoi)", "papaya", "guava", "macadamia nuts"],
            traditional_dishes=["poke", "lau lau", "poi (taro)", "plate lunch", "kalua pork", "acai bowls"],
            dietary_notes="Pacific Islander + Asian + American fusion. Taro is resistant starch.",
            telomere_relevant_foods=["poke (fish omega-3)", "taro (resistant starch)", "tropical fruits (vitamin C)", "seaweed"],
        ),
        "New York": StateProfile(
            name="New York",
            staple_foods=["wheat (bagels, bread, pizza)", "pasta"],
            proteins=["diverse — NYC is world's most diverse food city", "deli meats", "seafood"],
            vegetables=["diverse"],
            fruits=["apples (upstate)", "berries"],
            traditional_dishes=["pizza", "bagels", "cheesecake", "pastrami sandwich", "diverse ethnic cuisines"],
            dietary_notes="Most diverse food city on earth — access to every cuisine.",
        ),
    },
)

_MEXICO = CountryProfile(
    name="Mexico",
    region_id="central_america",
    staple_foods=["corn (nixtamalized — tortillas)", "beans", "rice"],
    proteins=["beans (primary)", "chicken", "pork", "fish (coastal)", "eggs", "cheese (queso fresco)"],
    vegetables=["squash", "tomatoes", "chili peppers", "nopal (prickly pear cactus)", "avocado", "jicama"],
    fruits=["papaya", "mango", "guava", "lime", "prickly pear fruit"],
    spices=["chili peppers (chipotle, ancho, guajillo)", "cumin", "oregano", "cinnamon", "epazote"],
    traditional_dishes=["tacos", "mole", "tamales", "enchiladas", "pozole", "chilaquiles"],
    dietary_notes="Corn+beans = complete protein. Nixtamalization releases niacin, increases calcium.",
    common_deficiencies=["iron", "zinc", "vitamin_a"],
    states={
        "Oaxaca": StateProfile(
            name="Oaxaca",
            staple_foods=["corn (native varieties — blue, red, white)", "beans", "squash"],
            proteins=["beans", "chapulines (grasshoppers — 72% protein)", "chicken", "goat", "quesillo"],
            vegetables=["squash", "chili peppers", "herbs"],
            spices=["complex mole sauces (7 varieties)", "chocolate", "multiple chili types"],
            traditional_dishes=["mole (7 varieties)", "tlayudas", "chapulines", "mezcal"],
            dietary_notes="Land of Seven Moles. Chapulines (72% protein, sustainable). Heritage corn varieties.",
            telomere_relevant_foods=["heritage corn (anthocyanins in blue/purple)", "chapulines (protein)", "dark chocolate (flavanols)"],
        ),
        "Yucatan": StateProfile(
            name="Yucatan",
            staple_foods=["corn", "beans"],
            proteins=["turkey (native)", "pork (cochinita pibil)", "fish/seafood", "black beans"],
            vegetables=["chaya (tree spinach — superfood)", "habanero chili", "achiote/annatto"],
            fruits=["papaya", "citrus", "soursop"],
            spices=["achiote (annatto — antioxidant)", "habanero", "oregano"],
            traditional_dishes=["cochinita pibil", "papadzules", "sopa de lima", "poc chuc"],
            dietary_notes="Chaya is higher in protein, calcium, iron than spinach. Maya heritage cuisine.",
            telomere_relevant_foods=["chaya (nutrient-dense superfood)", "habanero (capsaicin)", "achiote (carotenoid antioxidant)"],
        ),
    },
)

_BRAZIL = CountryProfile(
    name="Brazil",
    region_id="south_america",
    staple_foods=["rice", "cassava/manioc", "corn", "wheat"],
    proteins=["beans (feijão — daily)", "beef", "chicken", "fish (Amazon/coastal)", "pork"],
    vegetables=["tomatoes", "onions", "peppers", "squash", "hearts of palm"],
    fruits=["açaí", "passion fruit", "guava", "cashew fruit", "camu camu", "mango", "papaya"],
    spices=["garlic", "onion", "cilantro", "bay leaf", "cumin"],
    traditional_dishes=["feijoada (black bean stew)", "arroz com feijão (rice and beans)", "churrasco", "pão de queijo", "moqueca"],
    dietary_notes="Daily rice-and-beans is complementary protein, high fiber. Açaí is powerful antioxidant. Brazilian dietary guidelines are internationally praised.",
    common_deficiencies=["iron", "vitamin_a", "vitamin_d (southern latitudes)"],
    states={
        "Amazonas": StateProfile(
            name="Amazonas",
            staple_foods=["cassava (farinha)", "açaí"],
            proteins=["river fish (pirarucu, tambaqui)", "Brazil nuts", "açaí"],
            fruits=["açaí", "cupuaçu", "camu camu (highest vitamin C)", "guaraná"],
            traditional_dishes=["açaí bowls", "tacacá (soup)", "tucupi"],
            dietary_notes="Brazil nuts — 1 nut = daily selenium. Açaí — highest ORAC antioxidant value.",
            telomere_relevant_foods=["açaí (anthocyanins)", "Brazil nuts (selenium — telomere maintenance)", "camu camu (vitamin C)"],
        ),
        "Bahia": StateProfile(
            name="Bahia",
            staple_foods=["cassava", "corn", "rice"],
            proteins=["black beans", "dried/salted meat", "fish (coastal)", "dried shrimp"],
            vegetables=["okra", "palm hearts"],
            fruits=["cashew fruit (5x vitamin C of orange)", "coconut"],
            spices=["dendê/palm oil (tocotrienols)", "coconut milk", "coriander"],
            traditional_dishes=["acarajé (black-eyed pea fritters)", "moqueca (fish stew)", "vatapá", "caruru"],
            dietary_notes="Afro-Brazilian cuisine. Dendê (red palm oil) is richest source of tocotrienols.",
            telomere_relevant_foods=["red palm oil (tocotrienols, beta-carotene)", "cashew fruit (vitamin C)", "black-eyed peas"],
        ),
        "Rio Grande do Sul": StateProfile(
            name="Rio Grande do Sul",
            staple_foods=["rice", "polenta (Italian heritage)", "bread"],
            proteins=["beef (gaucho churrasco)", "pork", "Italian sausages"],
            fruits=["grapes (wine — Serra Gaúcha)"],
            spices=["chimichurri", "garlic"],
            traditional_dishes=["churrasco (barbecue)", "chimarrão/yerba mate", "polenta"],
            dietary_notes="Yerba mate (polyphenols, saponins — anti-obesity, lipid-lowering). Extremely high red meat.",
            telomere_relevant_foods=["yerba mate (antioxidants)", "wine polyphenols"],
        ),
    },
)

_ARGENTINA = CountryProfile(
    name="Argentina",
    region_id="south_america",
    staple_foods=["wheat (bread, pasta, empanadas)", "potatoes", "corn"],
    proteins=["beef (asado — primary)", "lamb (Patagonia)", "chicken", "lentils"],
    vegetables=["tomato", "onion", "peppers", "squash"],
    fruits=["grapes (wine — Mendoza)", "citrus", "berries (Patagonia)"],
    spices=["chimichurri (parsley, garlic, oregano, chili)", "oregano", "cumin"],
    traditional_dishes=["asado (barbecue)", "empanadas", "milanesa", "dulce de leche", "mate"],
    dietary_notes="World's highest per-capita beef consumption. Mate (yerba mate) is daily ritual.",
    common_deficiencies=["vitamin_d", "folate"],
)

_PERU = CountryProfile(
    name="Peru",
    region_id="andean",
    staple_foods=["potatoes (thousands of varieties)", "quinoa", "corn", "rice"],
    proteins=["beans", "fish/seafood (ceviche)", "chicken", "guinea pig (cuy — Andes)", "quinoa"],
    vegetables=["corn", "peppers (ají)", "squash", "tomato"],
    fruits=["lucuma", "cherimoya", "passion fruit", "camu camu", "mango"],
    spices=["ají amarillo", "huacatay (black mint)", "cilantro", "cumin"],
    traditional_dishes=["ceviche", "lomo saltado", "causa", "ají de gallina", "pachamanca"],
    dietary_notes="Quinoa — complete protein, all essential amino acids. Thousands of potato varieties. Camu camu — highest vitamin C.",
    common_deficiencies=["iron", "vitamin_d (highlands)", "iodine"],
)

# ---------- OCEANIA ----------

_AUSTRALIA = CountryProfile(
    name="Australia",
    region_id="australian",
    staple_foods=["wheat (bread)", "rice", "potatoes"],
    proteins=["beef", "lamb", "chicken", "seafood", "kangaroo (lean)", "dairy"],
    vegetables=["diverse — Mediterranean climate", "bush tomato", "warrigal greens"],
    fruits=["Kakadu plum (3000mg vitamin C/100g — 50x oranges)", "quandong", "Davidson plum", "macadamia", "citrus", "berries"],
    spices=["lemon myrtle", "wattle seed", "bush tomato", "mountain pepper"],
    traditional_dishes=["meat pie", "vegemite on toast", "barramundi", "lamingtons", "pavlova"],
    dietary_notes="Aboriginal bush foods are exceptionally nutrient-dense. Kakadu plum = highest vitamin C of any fruit.",
    common_deficiencies=["vitamin_d", "iodine"],
)

_NEW_ZEALAND = CountryProfile(
    name="New Zealand",
    region_id="australian",
    staple_foods=["wheat", "potatoes", "kumara (sweet potato)"],
    proteins=["lamb", "beef", "seafood (green-lipped mussels — anti-inflammatory)", "dairy", "venison"],
    vegetables=["kumara", "silver beet", "pumpkin"],
    fruits=["kiwifruit (vitamin C, actinidin enzyme)", "feijoa", "boysenberry"],
    spices=["horopito (mountain pepper — antimicrobial)", "kawakawa"],
    traditional_dishes=["hangi (earth oven)", "lamb roast", "meat pie", "pavlova", "fish and chips"],
    dietary_notes="Green-lipped mussels contain unique omega-3 (ETA — anti-inflammatory). Kiwifruit is vitamin C powerhouse.",
    common_deficiencies=["vitamin_d", "iodine"],
)


# =========================================================================
# Region → countries / Country → states lookup tables
# =========================================================================

# All country profiles keyed by name
_ALL_COUNTRIES: dict[str, CountryProfile] = {
    p.name: p
    for p in [
        _INDIA, _PAKISTAN, _BANGLADESH, _SRI_LANKA, _NEPAL,
        _CHINA, _JAPAN, _SOUTH_KOREA,
        _THAILAND, _VIETNAM, _INDONESIA,
        _TURKEY, _IRAN, _SAUDI_ARABIA,
        _UK, _FRANCE, _GERMANY, _ITALY, _SPAIN, _GREECE,
        _NIGERIA, _ETHIOPIA, _KENYA, _SOUTH_AFRICA,
        _USA, _MEXICO, _BRAZIL, _ARGENTINA, _PERU,
        _AUSTRALIA, _NEW_ZEALAND,
    ]
}

REGION_COUNTRIES: dict[str, list[str]] = {
    "East Asia": ["China", "Japan", "South Korea"],
    "South Asia": ["India", "Pakistan", "Bangladesh", "Sri Lanka", "Nepal"],
    "Southeast Asia": ["Thailand", "Vietnam", "Indonesia"],
    "Central Asia": [],
    "Western Asia": ["Turkey", "Iran", "Saudi Arabia"],
    "Northern Europe": ["United Kingdom"],
    "Western Europe": ["France", "Germany"],
    "Eastern Europe": [],
    "Southern Europe": ["Italy", "Spain", "Greece"],
    "North Africa": [],
    "Sub-Saharan Africa": ["Nigeria", "Ethiopia", "Kenya", "South Africa"],
    "North America": ["United States"],
    "Central America": ["Mexico"],
    "South America": ["Brazil", "Argentina", "Peru"],
    "Oceania": ["Australia", "New Zealand"],
}

COUNTRY_STATES: dict[str, list[str]] = {
    name: sorted(cp.states.keys()) for name, cp in _ALL_COUNTRIES.items() if cp.states
}


# =========================================================================
# Public API functions
# =========================================================================

def resolve_region(
    frontend_region: str,
    *,
    country: str | None = None,
    state: str | None = None,
) -> str:
    """Resolve a frontend region label + optional country/state to a diet_advisor region_id.

    Priority: state override > country override > frontend map > pass-through.
    """
    # 1. State-level override (most specific)
    if country and state and country in _STATE_REGION_OVERRIDE:
        state_map = _STATE_REGION_OVERRIDE[country]
        if state in state_map:
            return state_map[state]

    # 2. Country-level override
    if country and country in _COUNTRY_REGION_OVERRIDE:
        return _COUNTRY_REGION_OVERRIDE[country]

    # 3. Frontend label → region_id mapping
    if frontend_region in FRONTEND_REGION_MAP:
        return FRONTEND_REGION_MAP[frontend_region]

    # 4. Pass-through (assume it's already a valid region_id)
    return frontend_region


def get_country_profile(country: str) -> CountryProfile | None:
    """Return the full dietary profile for a country, or ``None``."""
    return _ALL_COUNTRIES.get(country)


def get_state_profile(country: str, state: str) -> StateProfile | None:
    """Return the dietary profile for a state within a country, or ``None``."""
    cp = _ALL_COUNTRIES.get(country)
    if cp is None:
        return None
    return cp.states.get(state)


def list_countries_for_region(frontend_region: str) -> list[str]:
    """Return the list of countries available for a given frontend region label."""
    return REGION_COUNTRIES.get(frontend_region, [])


def list_states_for_country(country: str) -> list[str]:
    """Return the list of states with detailed profiles for a given country."""
    return COUNTRY_STATES.get(country, [])
