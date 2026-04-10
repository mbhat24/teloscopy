"""Nutrition agent for dietary recommendations based on genetic profiles.

The :class:`NutritionAgent` translates genetic risk profiles and telomere
health data into actionable, region-aware dietary plans.  It maintains an
internal database of telomere-protective foods mapped to geographic
regions and generates personalised meal plans that account for individual
dietary restrictions and nutritional gaps.
"""

from __future__ import annotations

import logging
from typing import Any

from .base import AgentMessage, AgentState, BaseAgent

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Regional food databases (simplified reference data)
# ---------------------------------------------------------------------------

_TELOMERE_PROTECTIVE_FOODS: dict[str, list[dict[str, Any]]] = {
    "global": [
        {"name": "Blueberries", "nutrients": ["anthocyanins", "vitamin_c"], "category": "fruit"},
        {"name": "Spinach", "nutrients": ["folate", "iron", "vitamin_K"], "category": "vegetable"},
        {
            "name": "Salmon",
            "nutrients": ["omega_3", "vitamin_d", "selenium"],
            "category": "protein",
        },
        {
            "name": "Walnuts",
            "nutrients": ["omega_3", "vitamin_e", "polyphenols"],
            "category": "nuts",
        },
        {"name": "Green tea", "nutrients": ["EGCG", "catechins"], "category": "beverage"},
        {"name": "Turmeric", "nutrients": ["curcumin"], "category": "spice"},
        {"name": "Lentils", "nutrients": ["folate", "iron", "fiber"], "category": "legume"},
        {
            "name": "Dark chocolate (70%+)",
            "nutrients": ["flavonoids", "magnesium"],
            "category": "treat",
        },
        {
            "name": "Broccoli",
            "nutrients": ["sulforaphane", "vitamin_c", "fiber"],
            "category": "vegetable",
        },
        {
            "name": "Pomegranate",
            "nutrients": ["punicalagins", "vitamin_c", "polyphenols"],
            "category": "fruit",
        },
        {
            "name": "Garlic",
            "nutrients": ["allicin", "selenium", "vitamin_c"],
            "category": "vegetable",
        },
        {"name": "Flaxseeds", "nutrients": ["omega_3", "lignans", "fiber"], "category": "seed"},
        {
            "name": "Kale",
            "nutrients": ["vitamin_K", "vitamin_c", "lutein"],
            "category": "vegetable",
        },
        {"name": "Almonds", "nutrients": ["vitamin_e", "magnesium", "fiber"], "category": "nuts"},
        {"name": "Oats", "nutrients": ["beta_glucans", "fiber", "manganese"], "category": "grain"},
        {"name": "Ginger", "nutrients": ["gingerol", "manganese"], "category": "spice"},
    ],
    "mediterranean": [
        {
            "name": "Extra virgin olive oil",
            "nutrients": ["oleic_acid", "polyphenols"],
            "category": "fat",
        },
        {
            "name": "Sardines",
            "nutrients": ["omega_3", "calcium", "vitamin_d"],
            "category": "protein",
        },
        {"name": "Tomatoes", "nutrients": ["lycopene", "vitamin_c"], "category": "vegetable"},
        {"name": "Red grapes", "nutrients": ["resveratrol", "anthocyanins"], "category": "fruit"},
        {"name": "Chickpeas", "nutrients": ["folate", "fiber", "zinc"], "category": "legume"},
        {"name": "Figs", "nutrients": ["calcium", "potassium", "fiber"], "category": "fruit"},
        {
            "name": "Red bell pepper",
            "nutrients": ["vitamin_c", "beta_carotene", "lutein"],
            "category": "vegetable",
        },
        {
            "name": "Artichokes",
            "nutrients": ["fiber", "folate", "magnesium"],
            "category": "vegetable",
        },
        {"name": "Mackerel", "nutrients": ["omega_3", "B12", "selenium"], "category": "protein"},
        {
            "name": "Greek yoghurt",
            "nutrients": ["probiotics", "calcium", "B12"],
            "category": "dairy",
        },
        {
            "name": "Capers",
            "nutrients": ["quercetin", "rutin", "kaempferol"],
            "category": "vegetable",
        },
        {
            "name": "Arugula",
            "nutrients": ["vitamin_K", "folate", "sulforaphane"],
            "category": "vegetable",
        },
        {
            "name": "Pine nuts",
            "nutrients": ["manganese", "vitamin_e", "magnesium"],
            "category": "nuts",
        },
        {
            "name": "Oregano",
            "nutrients": ["polyphenols", "vitamin_K", "manganese"],
            "category": "spice",
        },
        {
            "name": "Mediterranean pomegranate",
            "nutrients": ["punicalagins", "vitamin_c", "ellagic_acid"],
            "category": "fruit",
        },
    ],
    "east_asian": [
        {"name": "Edamame", "nutrients": ["isoflavones", "folate"], "category": "legume"},
        {
            "name": "Matcha",
            "nutrients": ["EGCG", "L_theanine", "catechins"],
            "category": "beverage",
        },
        {
            "name": "Seaweed (nori)",
            "nutrients": ["iodine", "selenium", "omega_3"],
            "category": "vegetable",
        },
        {"name": "Miso", "nutrients": ["probiotics", "isoflavones"], "category": "fermented"},
        {
            "name": "Sweet potato",
            "nutrients": ["beta_carotene", "vitamin_c", "fiber"],
            "category": "vegetable",
        },
        {
            "name": "Shiitake mushrooms",
            "nutrients": ["vitamin_d", "selenium", "beta_glucans"],
            "category": "vegetable",
        },
        {
            "name": "Goji berries",
            "nutrients": ["beta_carotene", "vitamin_c", "zinc"],
            "category": "fruit",
        },
        {
            "name": "Bok choy",
            "nutrients": ["vitamin_c", "vitamin_K", "calcium"],
            "category": "vegetable",
        },
        {"name": "Tofu", "nutrients": ["isoflavones", "calcium", "iron"], "category": "protein"},
        {
            "name": "Black sesame seeds",
            "nutrients": ["calcium", "iron", "lignans"],
            "category": "seed",
        },
        {
            "name": "Bitter melon",
            "nutrients": ["vitamin_c", "folate", "fiber"],
            "category": "vegetable",
        },
        {
            "name": "Lotus root",
            "nutrients": ["vitamin_c", "fiber", "potassium"],
            "category": "vegetable",
        },
        {
            "name": "Chrysanthemum tea",
            "nutrients": ["flavonoids", "vitamin_c"],
            "category": "beverage",
        },
        {
            "name": "Chinese red dates (jujube)",
            "nutrients": ["vitamin_c", "potassium", "iron"],
            "category": "fruit",
        },
        {
            "name": "Wood ear mushroom",
            "nutrients": ["iron", "fiber", "potassium"],
            "category": "vegetable",
        },
    ],
    "south_asian": [
        {"name": "Turmeric (haldi)", "nutrients": ["curcumin"], "category": "spice"},
        {
            "name": "Moringa leaves",
            "nutrients": ["vitamin_c", "iron", "calcium"],
            "category": "vegetable",
        },
        {
            "name": "Amla (Indian gooseberry)",
            "nutrients": ["vitamin_c", "polyphenols"],
            "category": "fruit",
        },
        {"name": "Mung dal", "nutrients": ["folate", "fiber"], "category": "legume"},
        {
            "name": "Fenugreek seeds",
            "nutrients": ["fiber", "iron", "manganese"],
            "category": "spice",
        },
        {
            "name": "Curd (yoghurt)",
            "nutrients": ["probiotics", "calcium", "B12"],
            "category": "dairy",
        },
        {"name": "Ashwagandha tea", "nutrients": ["iron", "polyphenols"], "category": "beverage"},
        {
            "name": "Pomegranate (anar)",
            "nutrients": ["punicalagins", "vitamin_c"],
            "category": "fruit",
        },
        {
            "name": "Mustard greens (sarson)",
            "nutrients": ["vitamin_K", "vitamin_c", "folate"],
            "category": "vegetable",
        },
        {
            "name": "Flaxseeds (alsi)",
            "nutrients": ["omega_3", "lignans", "fiber"],
            "category": "seed",
        },
        {
            "name": "Drumstick pods (sahjan)",
            "nutrients": ["vitamin_c", "calcium", "iron"],
            "category": "vegetable",
        },
        {
            "name": "Jackfruit (kathal)",
            "nutrients": ["vitamin_c", "potassium", "magnesium"],
            "category": "fruit",
        },
        {
            "name": "Curry leaves",
            "nutrients": ["iron", "calcium", "beta_carotene"],
            "category": "spice",
        },
        {
            "name": "Black chickpeas (kala chana)",
            "nutrients": ["folate", "iron", "fiber"],
            "category": "legume",
        },
        {
            "name": "Coconut (nariyal)",
            "nutrients": ["manganese", "copper", "selenium"],
            "category": "fruit",
        },
    ],
    "americas": [
        {
            "name": "Avocado",
            "nutrients": ["oleic_acid", "folate", "potassium"],
            "category": "fruit",
        },
        {
            "name": "Black beans",
            "nutrients": ["anthocyanins", "folate", "fiber"],
            "category": "legume",
        },
        {"name": "Quinoa", "nutrients": ["magnesium", "folate"], "category": "grain"},
        {"name": "Chia seeds", "nutrients": ["omega_3", "fiber", "calcium"], "category": "seed"},
        {"name": "Acai berries", "nutrients": ["anthocyanins", "omega_3"], "category": "fruit"},
        {
            "name": "Wild salmon",
            "nutrients": ["omega_3", "astaxanthin", "vitamin_d"],
            "category": "protein",
        },
        {"name": "Pecans", "nutrients": ["manganese", "copper", "zinc"], "category": "nuts"},
        {
            "name": "Cranberries",
            "nutrients": ["vitamin_c", "quercetin", "anthocyanins"],
            "category": "fruit",
        },
        {
            "name": "Sweet potato",
            "nutrients": ["beta_carotene", "vitamin_c", "fiber"],
            "category": "vegetable",
        },
        {
            "name": "Pumpkin seeds",
            "nutrients": ["zinc", "magnesium", "omega_3"],
            "category": "seed",
        },
        {"name": "Turkey", "nutrients": ["selenium", "B12", "zinc"], "category": "protein"},
        {
            "name": "Cacao nibs",
            "nutrients": ["flavonoids", "magnesium", "iron"],
            "category": "treat",
        },
        {
            "name": "Nopal cactus",
            "nutrients": ["fiber", "vitamin_c", "magnesium"],
            "category": "vegetable",
        },
        {
            "name": "Brazil nuts",
            "nutrients": ["selenium", "magnesium", "vitamin_e"],
            "category": "nuts",
        },
        {
            "name": "Papaya",
            "nutrients": ["vitamin_c", "folate", "beta_carotene"],
            "category": "fruit",
        },
    ],
    "african": [
        {
            "name": "Baobab fruit",
            "nutrients": ["vitamin_c", "fiber", "calcium"],
            "category": "fruit",
        },
        {
            "name": "Moringa",
            "nutrients": ["vitamin_c", "iron", "calcium"],
            "category": "vegetable",
        },
        {"name": "Teff", "nutrients": ["iron", "calcium", "fiber"], "category": "grain"},
        {
            "name": "Cowpeas (black-eyed peas)",
            "nutrients": ["folate", "iron", "fiber"],
            "category": "legume",
        },
        {"name": "Okra", "nutrients": ["folate", "vitamin_c", "fiber"], "category": "vegetable"},
        {
            "name": "Hibiscus tea",
            "nutrients": ["anthocyanins", "vitamin_c"],
            "category": "beverage",
        },
        {
            "name": "Amaranth greens",
            "nutrients": ["iron", "calcium", "vitamin_c"],
            "category": "vegetable",
        },
        {
            "name": "Watermelon",
            "nutrients": ["lycopene", "vitamin_c", "potassium"],
            "category": "fruit",
        },
        {
            "name": "Groundnuts (peanuts)",
            "nutrients": ["folate", "magnesium", "resveratrol"],
            "category": "nuts",
        },
        {"name": "Millet", "nutrients": ["magnesium", "manganese", "fiber"], "category": "grain"},
        {"name": "Rooibos tea", "nutrients": ["polyphenols", "quercetin"], "category": "beverage"},
        {"name": "Tamarind", "nutrients": ["magnesium", "potassium", "iron"], "category": "fruit"},
        {
            "name": "Red palm oil",
            "nutrients": ["beta_carotene", "vitamin_e", "CoQ10"],
            "category": "fat",
        },
        {
            "name": "Cassava leaves",
            "nutrients": ["vitamin_c", "folate", "iron"],
            "category": "vegetable",
        },
        {"name": "Fonio", "nutrients": ["iron", "zinc", "fiber"], "category": "grain"},
    ],
    "nordic": [
        {
            "name": "Wild blueberries",
            "nutrients": ["anthocyanins", "vitamin_c", "manganese"],
            "category": "fruit",
        },
        {"name": "Herring", "nutrients": ["omega_3", "vitamin_d", "B12"], "category": "protein"},
        {
            "name": "Lingonberries",
            "nutrients": ["quercetin", "vitamin_c", "manganese"],
            "category": "fruit",
        },
        {
            "name": "Rye bread",
            "nutrients": ["fiber", "manganese", "selenium"],
            "category": "grain",
        },
        {
            "name": "Cloudberries",
            "nutrients": ["vitamin_c", "ellagic_acid", "vitamin_e"],
            "category": "fruit",
        },
        {
            "name": "Rapeseed oil",
            "nutrients": ["omega_3", "vitamin_e", "oleic_acid"],
            "category": "fat",
        },
        {
            "name": "Oat porridge",
            "nutrients": ["beta_glucans", "fiber", "manganese"],
            "category": "grain",
        },
        {
            "name": "Cabbage",
            "nutrients": ["vitamin_c", "vitamin_K", "sulforaphane"],
            "category": "vegetable",
        },
        {"name": "Dill", "nutrients": ["manganese", "vitamin_c", "folate"], "category": "spice"},
        {
            "name": "Smoked mackerel",
            "nutrients": ["omega_3", "B12", "vitamin_d"],
            "category": "protein",
        },
        {
            "name": "Sea buckthorn",
            "nutrients": ["vitamin_c", "vitamin_e", "omega_3"],
            "category": "fruit",
        },
        {
            "name": "Root vegetables (turnip)",
            "nutrients": ["vitamin_c", "fiber", "potassium"],
            "category": "vegetable",
        },
        {"name": "Elk meat", "nutrients": ["iron", "B12", "zinc"], "category": "protein"},
        {
            "name": "Fermented cabbage",
            "nutrients": ["probiotics", "vitamin_c", "vitamin_K"],
            "category": "fermented",
        },
        {
            "name": "Beetroot",
            "nutrients": ["folate", "manganese", "potassium"],
            "category": "vegetable",
        },
    ],
    "eastern_european": [
        {
            "name": "Buckwheat (kasha)",
            "nutrients": ["rutin", "magnesium", "manganese"],
            "category": "grain",
        },
        {
            "name": "Sauerkraut",
            "nutrients": ["probiotics", "vitamin_c", "vitamin_K"],
            "category": "fermented",
        },
        {
            "name": "Beetroot (borscht)",
            "nutrients": ["folate", "manganese", "potassium"],
            "category": "vegetable",
        },
        {
            "name": "Sunflower seeds",
            "nutrients": ["vitamin_e", "selenium", "magnesium"],
            "category": "seed",
        },
        {"name": "Kefir", "nutrients": ["probiotics", "calcium", "B12"], "category": "dairy"},
        {
            "name": "Horseradish",
            "nutrients": ["vitamin_c", "folate", "potassium"],
            "category": "spice",
        },
        {
            "name": "Black currants",
            "nutrients": ["vitamin_c", "anthocyanins", "manganese"],
            "category": "fruit",
        },
        {
            "name": "Dried mushrooms (porcini)",
            "nutrients": ["vitamin_d", "selenium", "B12"],
            "category": "vegetable",
        },
        {"name": "Poppy seeds", "nutrients": ["calcium", "manganese", "zinc"], "category": "seed"},
        {
            "name": "Fermented pickles",
            "nutrients": ["probiotics", "vitamin_K"],
            "category": "fermented",
        },
        {
            "name": "Rye grain",
            "nutrients": ["fiber", "manganese", "selenium"],
            "category": "grain",
        },
        {
            "name": "Sour cherries",
            "nutrients": ["anthocyanins", "vitamin_c", "quercetin"],
            "category": "fruit",
        },
        {
            "name": "Walnuts",
            "nutrients": ["omega_3", "polyphenols", "vitamin_e"],
            "category": "nuts",
        },
        {
            "name": "Garlic",
            "nutrients": ["allicin", "selenium", "vitamin_c"],
            "category": "vegetable",
        },
        {
            "name": "Red cabbage",
            "nutrients": ["anthocyanins", "vitamin_c", "vitamin_K"],
            "category": "vegetable",
        },
    ],
    "middle_eastern": [
        {"name": "Dates", "nutrients": ["potassium", "magnesium", "fiber"], "category": "fruit"},
        {"name": "Tahini", "nutrients": ["calcium", "iron", "zinc"], "category": "seed"},
        {
            "name": "Sumac",
            "nutrients": ["polyphenols", "vitamin_c", "quercetin"],
            "category": "spice",
        },
        {
            "name": "Za'atar herb blend",
            "nutrients": ["flavonoids", "iron", "calcium"],
            "category": "spice",
        },
        {
            "name": "Pomegranate",
            "nutrients": ["punicalagins", "vitamin_c", "ellagic_acid"],
            "category": "fruit",
        },
        {"name": "Lamb liver", "nutrients": ["B12", "iron", "folate"], "category": "protein"},
        {
            "name": "Bulgur wheat",
            "nutrients": ["fiber", "manganese", "magnesium"],
            "category": "grain",
        },
        {
            "name": "Pistachios",
            "nutrients": ["vitamin_e", "potassium", "lutein"],
            "category": "nuts",
        },
        {"name": "Freekeh", "nutrients": ["fiber", "zinc", "selenium"], "category": "grain"},
        {"name": "Labneh", "nutrients": ["probiotics", "calcium", "B12"], "category": "dairy"},
        {
            "name": "Black seed (nigella)",
            "nutrients": ["selenium", "iron", "zinc"],
            "category": "seed",
        },
        {
            "name": "Parsley (tabouleh)",
            "nutrients": ["vitamin_c", "vitamin_K", "folate"],
            "category": "vegetable",
        },
        {"name": "Mint", "nutrients": ["manganese", "vitamin_c", "folate"], "category": "spice"},
        {"name": "Fava beans", "nutrients": ["folate", "iron", "fiber"], "category": "legume"},
        {
            "name": "Olives",
            "nutrients": ["oleic_acid", "vitamin_e", "polyphenols"],
            "category": "fruit",
        },
    ],
    "southeast_asian": [
        {
            "name": "Turmeric (kunyit)",
            "nutrients": ["curcumin", "manganese", "iron"],
            "category": "spice",
        },
        {
            "name": "Lemongrass",
            "nutrients": ["vitamin_c", "folate", "manganese"],
            "category": "spice",
        },
        {"name": "Galangal", "nutrients": ["gingerol", "vitamin_c", "iron"], "category": "spice"},
        {
            "name": "Moringa (malunggay)",
            "nutrients": ["vitamin_c", "iron", "calcium"],
            "category": "vegetable",
        },
        {
            "name": "Pandan leaves",
            "nutrients": ["beta_carotene", "vitamin_c"],
            "category": "vegetable",
        },
        {
            "name": "Tempeh",
            "nutrients": ["probiotics", "isoflavones", "manganese"],
            "category": "fermented",
        },
        {"name": "Coconut oil", "nutrients": ["oleic_acid", "vitamin_e"], "category": "fat"},
        {
            "name": "Papaya",
            "nutrients": ["vitamin_c", "folate", "beta_carotene"],
            "category": "fruit",
        },
        {
            "name": "Mangosteen",
            "nutrients": ["vitamin_c", "folate", "manganese"],
            "category": "fruit",
        },
        {
            "name": "Water spinach (kangkung)",
            "nutrients": ["iron", "vitamin_c", "vitamin_K"],
            "category": "vegetable",
        },
        {
            "name": "Snake fruit (salak)",
            "nutrients": ["beta_carotene", "vitamin_c"],
            "category": "fruit",
        },
        {
            "name": "Thai basil",
            "nutrients": ["vitamin_K", "manganese", "iron"],
            "category": "spice",
        },
        {
            "name": "Long beans",
            "nutrients": ["folate", "vitamin_c", "fiber"],
            "category": "legume",
        },
        {"name": "Tamarind", "nutrients": ["magnesium", "potassium", "iron"], "category": "fruit"},
        {
            "name": "Jackfruit",
            "nutrients": ["vitamin_c", "potassium", "magnesium"],
            "category": "fruit",
        },
    ],
    "japanese": [
        {
            "name": "Natto",
            "nutrients": ["vitamin_K", "probiotics", "isoflavones"],
            "category": "fermented",
        },
        {
            "name": "Matcha",
            "nutrients": ["EGCG", "L_theanine", "catechins"],
            "category": "beverage",
        },
        {
            "name": "Salmon roe (ikura)",
            "nutrients": ["omega_3", "astaxanthin", "vitamin_d"],
            "category": "protein",
        },
        {
            "name": "Shiso (perilla leaf)",
            "nutrients": ["omega_3", "vitamin_c", "iron"],
            "category": "vegetable",
        },
        {
            "name": "Daikon radish",
            "nutrients": ["vitamin_c", "fiber", "potassium"],
            "category": "vegetable",
        },
        {
            "name": "Sencha green tea",
            "nutrients": ["EGCG", "catechins", "vitamin_c"],
            "category": "beverage",
        },
        {
            "name": "Wakame seaweed",
            "nutrients": ["iodine", "magnesium", "folate"],
            "category": "vegetable",
        },
        {
            "name": "Soba noodles (buckwheat)",
            "nutrients": ["rutin", "manganese", "fiber"],
            "category": "grain",
        },
        {
            "name": "Umeboshi (pickled plum)",
            "nutrients": ["polyphenols", "iron"],
            "category": "fermented",
        },
        {
            "name": "Yuzu",
            "nutrients": ["vitamin_c", "hesperidin", "flavonoids"],
            "category": "fruit",
        },
        {
            "name": "Konbu (kelp)",
            "nutrients": ["iodine", "iron", "calcium"],
            "category": "vegetable",
        },
        {
            "name": "Japanese sweet potato",
            "nutrients": ["beta_carotene", "vitamin_c", "potassium"],
            "category": "vegetable",
        },
        {"name": "Azuki beans", "nutrients": ["folate", "iron", "fiber"], "category": "legume"},
        {
            "name": "Fresh ginger",
            "nutrients": ["gingerol", "manganese", "potassium"],
            "category": "spice",
        },
        {
            "name": "Maitake mushroom",
            "nutrients": ["vitamin_d", "beta_glucans", "selenium"],
            "category": "vegetable",
        },
    ],
    "korean": [
        {
            "name": "Kimchi",
            "nutrients": ["probiotics", "vitamin_c", "vitamin_K"],
            "category": "fermented",
        },
        {
            "name": "Gochugaru (red pepper flakes)",
            "nutrients": ["capsaicin", "vitamin_c", "beta_carotene"],
            "category": "spice",
        },
        {
            "name": "Perilla leaves (kkaennip)",
            "nutrients": ["omega_3", "iron", "vitamin_c"],
            "category": "vegetable",
        },
        {
            "name": "Doenjang (soybean paste)",
            "nutrients": ["probiotics", "isoflavones"],
            "category": "fermented",
        },
        {
            "name": "Korean pear",
            "nutrients": ["fiber", "vitamin_c", "potassium"],
            "category": "fruit",
        },
        {
            "name": "Korean ginseng tea",
            "nutrients": ["polyphenols", "iron", "zinc"],
            "category": "beverage",
        },
        {
            "name": "Barley tea (boricha)",
            "nutrients": ["selenium", "manganese"],
            "category": "beverage",
        },
        {
            "name": "Seaweed (miyeok)",
            "nutrients": ["iodine", "calcium", "iron"],
            "category": "vegetable",
        },
        {
            "name": "Black garlic",
            "nutrients": ["allicin", "polyphenols", "selenium"],
            "category": "vegetable",
        },
        {
            "name": "Jujubes (daechu)",
            "nutrients": ["vitamin_c", "potassium", "fiber"],
            "category": "fruit",
        },
        {
            "name": "Sesame leaves (kkae)",
            "nutrients": ["calcium", "iron", "vitamin_K"],
            "category": "vegetable",
        },
        {
            "name": "Mugwort (ssuk)",
            "nutrients": ["vitamin_c", "calcium", "iron"],
            "category": "vegetable",
        },
        {
            "name": "Cheonggukjang (fermented soy)",
            "nutrients": ["probiotics", "isoflavones", "vitamin_K"],
            "category": "fermented",
        },
        {
            "name": "Sweet potato (goguma)",
            "nutrients": ["beta_carotene", "vitamin_c", "fiber"],
            "category": "vegetable",
        },
        {
            "name": "Mung bean sprouts",
            "nutrients": ["vitamin_c", "folate", "fiber"],
            "category": "legume",
        },
    ],
    "caribbean": [
        {
            "name": "Callaloo",
            "nutrients": ["iron", "calcium", "vitamin_c"],
            "category": "vegetable",
        },
        {"name": "Ackee", "nutrients": ["folate", "potassium", "omega_3"], "category": "fruit"},
        {
            "name": "Scotch bonnet pepper",
            "nutrients": ["capsaicin", "vitamin_c"],
            "category": "spice",
        },
        {
            "name": "Breadfruit",
            "nutrients": ["fiber", "potassium", "vitamin_c"],
            "category": "fruit",
        },
        {"name": "Soursop", "nutrients": ["vitamin_c", "fiber", "magnesium"], "category": "fruit"},
        {
            "name": "Pigeon peas",
            "nutrients": ["folate", "fiber", "manganese"],
            "category": "legume",
        },
        {"name": "Guava", "nutrients": ["vitamin_c", "lycopene", "fiber"], "category": "fruit"},
        {
            "name": "Sorrel (hibiscus drink)",
            "nutrients": ["anthocyanins", "vitamin_c", "iron"],
            "category": "beverage",
        },
        {
            "name": "Dasheen (taro)",
            "nutrients": ["fiber", "potassium", "manganese"],
            "category": "vegetable",
        },
        {
            "name": "Plantain",
            "nutrients": ["potassium", "vitamin_c", "fiber"],
            "category": "fruit",
        },
        {
            "name": "Coconut water",
            "nutrients": ["potassium", "magnesium", "manganese"],
            "category": "beverage",
        },
        {
            "name": "Cassava",
            "nutrients": ["vitamin_c", "manganese", "fiber"],
            "category": "vegetable",
        },
        {
            "name": "Noni fruit",
            "nutrients": ["vitamin_c", "potassium", "polyphenols"],
            "category": "fruit",
        },
        {
            "name": "Cho cho (chayote)",
            "nutrients": ["folate", "vitamin_c", "fiber"],
            "category": "vegetable",
        },
        {
            "name": "Caribbean moringa",
            "nutrients": ["vitamin_c", "iron", "calcium"],
            "category": "vegetable",
        },
    ],
    "pacific_island": [
        {
            "name": "Taro",
            "nutrients": ["fiber", "potassium", "manganese"],
            "category": "vegetable",
        },
        {"name": "Coconut", "nutrients": ["manganese", "copper", "selenium"], "category": "fruit"},
        {
            "name": "Noni fruit",
            "nutrients": ["vitamin_c", "potassium", "polyphenols"],
            "category": "fruit",
        },
        {
            "name": "Breadfruit",
            "nutrients": ["fiber", "potassium", "vitamin_c"],
            "category": "fruit",
        },
        {
            "name": "Sweet potato (kumara)",
            "nutrients": ["beta_carotene", "vitamin_c", "fiber"],
            "category": "vegetable",
        },
        {
            "name": "Seaweed (limu)",
            "nutrients": ["iodine", "calcium", "iron"],
            "category": "vegetable",
        },
        {
            "name": "Wild-caught tuna",
            "nutrients": ["omega_3", "selenium", "B12"],
            "category": "protein",
        },
        {
            "name": "Papaya (pawpaw)",
            "nutrients": ["vitamin_c", "folate", "beta_carotene"],
            "category": "fruit",
        },
        {
            "name": "Cassava",
            "nutrients": ["vitamin_c", "manganese", "fiber"],
            "category": "vegetable",
        },
        {"name": "Ti leaf tea", "nutrients": ["polyphenols", "manganese"], "category": "beverage"},
        {
            "name": "Poi (fermented taro)",
            "nutrients": ["probiotics", "fiber", "potassium"],
            "category": "fermented",
        },
        {
            "name": "Macadamia nuts",
            "nutrients": ["manganese", "oleic_acid", "magnesium"],
            "category": "nuts",
        },
        {
            "name": "Pandanus fruit",
            "nutrients": ["beta_carotene", "vitamin_c"],
            "category": "fruit",
        },
        {
            "name": "Banana",
            "nutrients": ["potassium", "vitamin_c", "magnesium"],
            "category": "fruit",
        },
        {
            "name": "Yam",
            "nutrients": ["vitamin_c", "potassium", "manganese"],
            "category": "vegetable",
        },
    ],
    "central_asian": [
        {
            "name": "Dried apricots",
            "nutrients": ["beta_carotene", "potassium", "iron"],
            "category": "fruit",
        },
        {
            "name": "Pomegranate",
            "nutrients": ["punicalagins", "vitamin_c", "polyphenols"],
            "category": "fruit",
        },
        {"name": "Lamb", "nutrients": ["iron", "B12", "zinc"], "category": "protein"},
        {
            "name": "Walnuts",
            "nutrients": ["omega_3", "polyphenols", "vitamin_e"],
            "category": "nuts",
        },
        {"name": "Cumin", "nutrients": ["iron", "manganese", "calcium"], "category": "spice"},
        {
            "name": "Fermented mare milk (kumis)",
            "nutrients": ["probiotics", "calcium", "B12"],
            "category": "dairy",
        },
        {"name": "Chickpeas", "nutrients": ["folate", "fiber", "zinc"], "category": "legume"},
        {"name": "Barley", "nutrients": ["fiber", "selenium", "manganese"], "category": "grain"},
        {
            "name": "Dried mulberries",
            "nutrients": ["iron", "vitamin_c", "resveratrol"],
            "category": "fruit",
        },
        {"name": "Almonds", "nutrients": ["vitamin_e", "magnesium", "fiber"], "category": "nuts"},
        {
            "name": "Green tea",
            "nutrients": ["EGCG", "catechins", "L_theanine"],
            "category": "beverage",
        },
        {
            "name": "Carrots",
            "nutrients": ["beta_carotene", "fiber", "vitamin_K"],
            "category": "vegetable",
        },
        {
            "name": "Pumpkin",
            "nutrients": ["beta_carotene", "vitamin_c", "potassium"],
            "category": "vegetable",
        },
        {
            "name": "Rose hip tea",
            "nutrients": ["vitamin_c", "polyphenols", "vitamin_e"],
            "category": "beverage",
        },
        {"name": "Sesame seeds", "nutrients": ["calcium", "iron", "zinc"], "category": "seed"},
    ],
    "north_african": [
        {
            "name": "Argan oil",
            "nutrients": ["vitamin_e", "oleic_acid", "polyphenols"],
            "category": "fat",
        },
        {
            "name": "Whole grain couscous",
            "nutrients": ["fiber", "selenium", "manganese"],
            "category": "grain",
        },
        {"name": "Dates", "nutrients": ["potassium", "magnesium", "fiber"], "category": "fruit"},
        {
            "name": "Preserved lemons",
            "nutrients": ["vitamin_c", "flavonoids"],
            "category": "fruit",
        },
        {
            "name": "Harissa peppers",
            "nutrients": ["capsaicin", "vitamin_c", "vitamin_e"],
            "category": "spice",
        },
        {
            "name": "Ras el hanout spices",
            "nutrients": ["curcumin", "gingerol", "iron"],
            "category": "spice",
        },
        {
            "name": "Olives",
            "nutrients": ["oleic_acid", "vitamin_e", "polyphenols"],
            "category": "fruit",
        },
        {"name": "Almonds", "nutrients": ["vitamin_e", "magnesium", "fiber"], "category": "nuts"},
        {"name": "Chickpeas", "nutrients": ["folate", "fiber", "zinc"], "category": "legume"},
        {
            "name": "Mint tea",
            "nutrients": ["manganese", "flavonoids", "polyphenols"],
            "category": "beverage",
        },
        {"name": "Figs", "nutrients": ["calcium", "potassium", "fiber"], "category": "fruit"},
        {
            "name": "Sardines",
            "nutrients": ["omega_3", "calcium", "vitamin_d"],
            "category": "protein",
        },
        {
            "name": "Broad beans (foul)",
            "nutrients": ["folate", "iron", "fiber"],
            "category": "legume",
        },
        {
            "name": "Orange blossom tea",
            "nutrients": ["flavonoids", "polyphenols"],
            "category": "beverage",
        },
        {"name": "Barley", "nutrients": ["fiber", "selenium", "manganese"], "category": "grain"},
    ],
    "west_african": [
        {
            "name": "Red palm oil",
            "nutrients": ["beta_carotene", "vitamin_e", "CoQ10"],
            "category": "fat",
        },
        {
            "name": "Egusi seeds (melon seeds)",
            "nutrients": ["omega_3", "iron", "zinc"],
            "category": "seed",
        },
        {
            "name": "Bitter leaf",
            "nutrients": ["vitamin_c", "iron", "polyphenols"],
            "category": "vegetable",
        },
        {
            "name": "Plantain",
            "nutrients": ["potassium", "vitamin_c", "fiber"],
            "category": "fruit",
        },
        {
            "name": "Dawadawa (locust bean)",
            "nutrients": ["iron", "probiotics", "zinc"],
            "category": "fermented",
        },
        {
            "name": "Tiger nuts",
            "nutrients": ["fiber", "vitamin_e", "oleic_acid"],
            "category": "nuts",
        },
        {"name": "Shea butter", "nutrients": ["vitamin_e", "oleic_acid"], "category": "fat"},
        {
            "name": "Baobab fruit",
            "nutrients": ["vitamin_c", "fiber", "calcium"],
            "category": "fruit",
        },
        {
            "name": "Black-eyed peas",
            "nutrients": ["folate", "iron", "fiber"],
            "category": "legume",
        },
        {"name": "Millet", "nutrients": ["magnesium", "manganese", "fiber"], "category": "grain"},
        {"name": "Okra", "nutrients": ["folate", "vitamin_c", "fiber"], "category": "vegetable"},
        {"name": "Guinea fowl", "nutrients": ["B12", "iron", "selenium"], "category": "protein"},
        {
            "name": "Hibiscus (zobo drink)",
            "nutrients": ["anthocyanins", "vitamin_c", "iron"],
            "category": "beverage",
        },
        {
            "name": "Moringa leaves",
            "nutrients": ["vitamin_c", "iron", "calcium"],
            "category": "vegetable",
        },
        {
            "name": "Groundnuts",
            "nutrients": ["folate", "magnesium", "resveratrol"],
            "category": "nuts",
        },
    ],
    "east_african": [
        {"name": "Teff", "nutrients": ["iron", "calcium", "fiber"], "category": "grain"},
        {
            "name": "Injera (fermented teff)",
            "nutrients": ["probiotics", "iron", "calcium"],
            "category": "fermented",
        },
        {
            "name": "Berbere spice blend",
            "nutrients": ["capsaicin", "iron", "vitamin_c"],
            "category": "spice",
        },
        {
            "name": "Collard greens (gomen)",
            "nutrients": ["vitamin_K", "vitamin_c", "calcium"],
            "category": "vegetable",
        },
        {
            "name": "Lentils (misir)",
            "nutrients": ["folate", "iron", "fiber"],
            "category": "legume",
        },
        {
            "name": "Grilled meat (nyama choma)",
            "nutrients": ["iron", "B12", "zinc"],
            "category": "protein",
        },
        {
            "name": "Avocado",
            "nutrients": ["oleic_acid", "folate", "potassium"],
            "category": "fruit",
        },
        {
            "name": "Sukuma wiki (kale)",
            "nutrients": ["vitamin_K", "vitamin_c", "lutein"],
            "category": "vegetable",
        },
        {
            "name": "Amaranth grain",
            "nutrients": ["iron", "calcium", "manganese"],
            "category": "grain",
        },
        {
            "name": "Chai spiced tea",
            "nutrients": ["polyphenols", "gingerol", "manganese"],
            "category": "beverage",
        },
        {
            "name": "Mango",
            "nutrients": ["vitamin_c", "beta_carotene", "folate"],
            "category": "fruit",
        },
        {"name": "Kidney beans", "nutrients": ["fiber", "iron", "folate"], "category": "legume"},
        {
            "name": "Papaya",
            "nutrients": ["vitamin_c", "folate", "beta_carotene"],
            "category": "fruit",
        },
        {
            "name": "Pumpkin leaves",
            "nutrients": ["iron", "vitamin_c", "calcium"],
            "category": "vegetable",
        },
        {
            "name": "Ethiopian coffee",
            "nutrients": ["chlorogenic_acid", "polyphenols", "magnesium"],
            "category": "beverage",
        },
    ],
    "andean": [
        {"name": "Quinoa", "nutrients": ["magnesium", "folate", "manganese"], "category": "grain"},
        {
            "name": "Maca root",
            "nutrients": ["iron", "copper", "manganese"],
            "category": "vegetable",
        },
        {
            "name": "Camu camu",
            "nutrients": ["vitamin_c", "flavonoids", "ellagic_acid"],
            "category": "fruit",
        },
        {
            "name": "Purple corn (maiz morado)",
            "nutrients": ["anthocyanins", "fiber"],
            "category": "grain",
        },
        {"name": "Lucuma", "nutrients": ["beta_carotene", "iron", "zinc"], "category": "fruit"},
        {
            "name": "Raw cacao",
            "nutrients": ["flavonoids", "magnesium", "iron"],
            "category": "treat",
        },
        {"name": "Chia seeds", "nutrients": ["omega_3", "fiber", "calcium"], "category": "seed"},
        {
            "name": "Sacha inchi seeds",
            "nutrients": ["omega_3", "vitamin_e", "zinc"],
            "category": "seed",
        },
        {"name": "Amaranth", "nutrients": ["iron", "calcium", "manganese"], "category": "grain"},
        {
            "name": "Purple potato",
            "nutrients": ["anthocyanins", "potassium", "vitamin_c"],
            "category": "vegetable",
        },
        {"name": "Kiwicha", "nutrients": ["iron", "calcium", "fiber"], "category": "grain"},
        {
            "name": "Goldenberry (aguaymanto)",
            "nutrients": ["vitamin_c", "beta_carotene", "polyphenols"],
            "category": "fruit",
        },
        {
            "name": "Yacon",
            "nutrients": ["fiber", "potassium", "polyphenols"],
            "category": "vegetable",
        },
        {
            "name": "Tarwi (Andean lupin)",
            "nutrients": ["folate", "zinc", "manganese"],
            "category": "legume",
        },
        {
            "name": "Coca leaf tea",
            "nutrients": ["flavonoids", "iron", "calcium"],
            "category": "beverage",
        },
    ],
    "australian": [
        {
            "name": "Macadamia nuts",
            "nutrients": ["manganese", "oleic_acid", "magnesium"],
            "category": "nuts",
        },
        {
            "name": "Kakadu plum",
            "nutrients": ["vitamin_c", "ellagic_acid", "folate"],
            "category": "fruit",
        },
        {
            "name": "Lemon myrtle",
            "nutrients": ["lutein", "vitamin_c", "calcium"],
            "category": "spice",
        },
        {"name": "Kangaroo meat", "nutrients": ["iron", "B12", "zinc"], "category": "protein"},
        {"name": "Wattleseed", "nutrients": ["magnesium", "iron", "zinc"], "category": "seed"},
        {"name": "Davidson plum", "nutrients": ["anthocyanins", "vitamin_c"], "category": "fruit"},
        {
            "name": "Quandong",
            "nutrients": ["vitamin_c", "folate", "magnesium"],
            "category": "fruit",
        },
        {
            "name": "Bush tomato",
            "nutrients": ["vitamin_c", "selenium", "iron"],
            "category": "fruit",
        },
        {
            "name": "Warrigal greens",
            "nutrients": ["vitamin_c", "magnesium", "manganese"],
            "category": "vegetable",
        },
        {
            "name": "Finger lime",
            "nutrients": ["vitamin_c", "folate", "potassium"],
            "category": "fruit",
        },
        {"name": "Barramundi", "nutrients": ["omega_3", "selenium", "B12"], "category": "protein"},
        {
            "name": "Tasmanian pepperberry",
            "nutrients": ["polyphenols", "manganese", "iron"],
            "category": "spice",
        },
        {"name": "Manuka honey", "nutrients": ["polyphenols", "manganese"], "category": "treat"},
        {
            "name": "Green-lipped mussel",
            "nutrients": ["omega_3", "iron", "B12"],
            "category": "protein",
        },
        {"name": "Feijoa", "nutrients": ["vitamin_c", "fiber", "folate"], "category": "fruit"},
    ],
    "british": [
        {
            "name": "Blackberries",
            "nutrients": ["vitamin_c", "anthocyanins", "fiber"],
            "category": "fruit",
        },
        {
            "name": "Watercress",
            "nutrients": ["vitamin_K", "vitamin_c", "iron"],
            "category": "vegetable",
        },
        {
            "name": "Oat porridge",
            "nutrients": ["beta_glucans", "fiber", "manganese"],
            "category": "grain",
        },
        {"name": "Mackerel", "nutrients": ["omega_3", "B12", "vitamin_d"], "category": "protein"},
        {
            "name": "Blackcurrants",
            "nutrients": ["vitamin_c", "anthocyanins", "manganese"],
            "category": "fruit",
        },
        {
            "name": "Brussels sprouts",
            "nutrients": ["sulforaphane", "vitamin_K", "vitamin_c"],
            "category": "vegetable",
        },
        {
            "name": "Aged cheddar cheese",
            "nutrients": ["calcium", "B12", "vitamin_K"],
            "category": "dairy",
        },
        {
            "name": "Garden peas",
            "nutrients": ["folate", "vitamin_c", "fiber"],
            "category": "legume",
        },
        {
            "name": "Elderberries",
            "nutrients": ["anthocyanins", "vitamin_c", "quercetin"],
            "category": "fruit",
        },
        {
            "name": "Stinging nettle tea",
            "nutrients": ["iron", "calcium", "vitamin_c"],
            "category": "beverage",
        },
        {
            "name": "Leeks",
            "nutrients": ["vitamin_K", "manganese", "folate"],
            "category": "vegetable",
        },
        {
            "name": "Kippers (smoked herring)",
            "nutrients": ["omega_3", "vitamin_d", "B12"],
            "category": "protein",
        },
        {
            "name": "Gooseberries",
            "nutrients": ["vitamin_c", "fiber", "manganese"],
            "category": "fruit",
        },
        {
            "name": "Walnuts",
            "nutrients": ["omega_3", "polyphenols", "vitamin_e"],
            "category": "nuts",
        },
        {
            "name": "Sage",
            "nutrients": ["vitamin_K", "manganese", "polyphenols"],
            "category": "spice",
        },
    ],
    "german": [
        {
            "name": "Sauerkraut",
            "nutrients": ["probiotics", "vitamin_c", "vitamin_K"],
            "category": "fermented",
        },
        {
            "name": "Linseed (flaxseed)",
            "nutrients": ["omega_3", "lignans", "fiber"],
            "category": "seed",
        },
        {
            "name": "Pumpernickel bread",
            "nutrients": ["fiber", "manganese", "selenium"],
            "category": "grain",
        },
        {"name": "Quark", "nutrients": ["calcium", "B12", "probiotics"], "category": "dairy"},
        {
            "name": "Walnuts",
            "nutrients": ["omega_3", "polyphenols", "vitamin_e"],
            "category": "nuts",
        },
        {
            "name": "Red cabbage",
            "nutrients": ["anthocyanins", "vitamin_c", "vitamin_K"],
            "category": "vegetable",
        },
        {"name": "Spelt", "nutrients": ["manganese", "fiber", "magnesium"], "category": "grain"},
        {
            "name": "Horseradish",
            "nutrients": ["vitamin_c", "folate", "potassium"],
            "category": "spice",
        },
        {
            "name": "Asparagus",
            "nutrients": ["folate", "vitamin_K", "vitamin_c"],
            "category": "vegetable",
        },
        {
            "name": "Beetroot",
            "nutrients": ["folate", "manganese", "potassium"],
            "category": "vegetable",
        },
        {"name": "Trout", "nutrients": ["omega_3", "B12", "selenium"], "category": "protein"},
        {
            "name": "Pumpkin seed oil",
            "nutrients": ["zinc", "magnesium", "vitamin_e"],
            "category": "fat",
        },
        {
            "name": "Black elderberry",
            "nutrients": ["vitamin_c", "anthocyanins", "quercetin"],
            "category": "fruit",
        },
        {
            "name": "Rapeseed oil",
            "nutrients": ["omega_3", "vitamin_e", "oleic_acid"],
            "category": "fat",
        },
        {
            "name": "Chamomile tea",
            "nutrients": ["apigenin", "flavonoids", "manganese"],
            "category": "beverage",
        },
    ],
    "french": [
        {
            "name": "Red wine (moderate)",
            "nutrients": ["resveratrol", "polyphenols"],
            "category": "beverage",
        },
        {
            "name": "Walnuts",
            "nutrients": ["omega_3", "polyphenols", "vitamin_e"],
            "category": "nuts",
        },
        {
            "name": "Artichokes",
            "nutrients": ["fiber", "folate", "magnesium"],
            "category": "vegetable",
        },
        {
            "name": "Roquefort cheese",
            "nutrients": ["probiotics", "calcium", "vitamin_K"],
            "category": "dairy",
        },
        {
            "name": "Lentils (Le Puy)",
            "nutrients": ["folate", "iron", "fiber"],
            "category": "legume",
        },
        {"name": "Lavender honey", "nutrients": ["polyphenols", "manganese"], "category": "treat"},
        {
            "name": "Chestnuts",
            "nutrients": ["vitamin_c", "manganese", "copper"],
            "category": "nuts",
        },
        {
            "name": "Shallots",
            "nutrients": ["quercetin", "allicin", "manganese"],
            "category": "vegetable",
        },
        {
            "name": "Prunes (Agen)",
            "nutrients": ["vitamin_K", "fiber", "potassium"],
            "category": "fruit",
        },
        {"name": "Endive", "nutrients": ["folate", "vitamin_K", "fiber"], "category": "vegetable"},
        {
            "name": "Sardines",
            "nutrients": ["omega_3", "calcium", "vitamin_d"],
            "category": "protein",
        },
        {"name": "Thyme", "nutrients": ["vitamin_c", "iron", "manganese"], "category": "spice"},
        {
            "name": "Olive oil (Provence)",
            "nutrients": ["oleic_acid", "polyphenols", "vitamin_e"],
            "category": "fat",
        },
        {
            "name": "Buckwheat (ble noir)",
            "nutrients": ["rutin", "magnesium", "manganese"],
            "category": "grain",
        },
        {
            "name": "Tarragon",
            "nutrients": ["manganese", "iron", "polyphenols"],
            "category": "spice",
        },
    ],
    "iberian": [
        {
            "name": "Olive oil",
            "nutrients": ["oleic_acid", "polyphenols", "vitamin_e"],
            "category": "fat",
        },
        {
            "name": "Marcona almonds",
            "nutrients": ["vitamin_e", "magnesium", "calcium"],
            "category": "nuts",
        },
        {
            "name": "Sardines",
            "nutrients": ["omega_3", "calcium", "vitamin_d"],
            "category": "protein",
        },
        {"name": "Saffron", "nutrients": ["manganese", "iron", "selenium"], "category": "spice"},
        {
            "name": "Garlic",
            "nutrients": ["allicin", "selenium", "vitamin_c"],
            "category": "vegetable",
        },
        {
            "name": "Quince (membrillo)",
            "nutrients": ["vitamin_c", "fiber", "copper"],
            "category": "fruit",
        },
        {
            "name": "Piquillo peppers",
            "nutrients": ["vitamin_c", "beta_carotene", "vitamin_e"],
            "category": "vegetable",
        },
        {
            "name": "Gazpacho vegetables",
            "nutrients": ["lycopene", "vitamin_c", "potassium"],
            "category": "vegetable",
        },
        {
            "name": "Blood orange",
            "nutrients": ["vitamin_c", "anthocyanins", "hesperidin"],
            "category": "fruit",
        },
        {
            "name": "Manchego cheese",
            "nutrients": ["calcium", "B12", "vitamin_K"],
            "category": "dairy",
        },
        {"name": "Octopus", "nutrients": ["B12", "iron", "selenium"], "category": "protein"},
        {"name": "Pinto beans", "nutrients": ["folate", "fiber", "iron"], "category": "legume"},
        {
            "name": "Port wine (moderate)",
            "nutrients": ["resveratrol", "polyphenols"],
            "category": "beverage",
        },
        {
            "name": "Chestnuts",
            "nutrients": ["vitamin_c", "manganese", "copper"],
            "category": "nuts",
        },
        {"name": "Piri piri pepper", "nutrients": ["capsaicin", "vitamin_c"], "category": "spice"},
    ],
    "indian_north": [
        {
            "name": "Ghee (clarified butter)",
            "nutrients": ["vitamin_K", "vitamin_e", "oleic_acid"],
            "category": "fat",
        },
        {
            "name": "Mustard oil",
            "nutrients": ["omega_3", "vitamin_e", "allicin"],
            "category": "fat",
        },
        {
            "name": "Amla (gooseberry)",
            "nutrients": ["vitamin_c", "polyphenols", "ellagic_acid"],
            "category": "fruit",
        },
        {"name": "Paneer", "nutrients": ["calcium", "B12", "selenium"], "category": "dairy"},
        {
            "name": "Brown basmati rice",
            "nutrients": ["fiber", "manganese", "selenium"],
            "category": "grain",
        },
        {
            "name": "Spinach (palak)",
            "nutrients": ["folate", "iron", "vitamin_K"],
            "category": "vegetable",
        },
        {
            "name": "Chickpeas (chole)",
            "nutrients": ["folate", "fiber", "zinc"],
            "category": "legume",
        },
        {
            "name": "Bitter gourd (karela)",
            "nutrients": ["vitamin_c", "folate", "fiber"],
            "category": "vegetable",
        },
        {
            "name": "Fenugreek (methi)",
            "nutrients": ["fiber", "iron", "manganese"],
            "category": "spice",
        },
        {
            "name": "Pomegranate (anardana)",
            "nutrients": ["punicalagins", "vitamin_c", "ellagic_acid"],
            "category": "fruit",
        },
        {"name": "Ashwagandha tea", "nutrients": ["iron", "polyphenols"], "category": "beverage"},
        {
            "name": "Jaggery (gur)",
            "nutrients": ["iron", "manganese", "potassium"],
            "category": "treat",
        },
        {
            "name": "Walnuts (akhrot)",
            "nutrients": ["omega_3", "polyphenols", "vitamin_e"],
            "category": "nuts",
        },
        {
            "name": "Ajwain (carom seeds)",
            "nutrients": ["manganese", "iron", "calcium"],
            "category": "spice",
        },
        {
            "name": "Sarson ka saag (mustard greens)",
            "nutrients": ["vitamin_K", "vitamin_c", "folate"],
            "category": "vegetable",
        },
    ],
    "indian_south": [
        {"name": "Coconut oil", "nutrients": ["oleic_acid", "vitamin_e"], "category": "fat"},
        {
            "name": "Curry leaves (kadi patta)",
            "nutrients": ["iron", "calcium", "beta_carotene"],
            "category": "spice",
        },
        {
            "name": "Drumstick (moringa pods)",
            "nutrients": ["vitamin_c", "calcium", "iron"],
            "category": "vegetable",
        },
        {"name": "Tamarind", "nutrients": ["magnesium", "potassium", "iron"], "category": "fruit"},
        {
            "name": "Ragi (finger millet)",
            "nutrients": ["calcium", "iron", "fiber"],
            "category": "grain",
        },
        {"name": "Curd rice", "nutrients": ["probiotics", "calcium", "B12"], "category": "dairy"},
        {
            "name": "Jackfruit",
            "nutrients": ["vitamin_c", "potassium", "magnesium"],
            "category": "fruit",
        },
        {
            "name": "Kokum",
            "nutrients": ["vitamin_c", "manganese", "polyphenols"],
            "category": "fruit",
        },
        {
            "name": "Urad dal (black gram)",
            "nutrients": ["folate", "iron", "fiber"],
            "category": "legume",
        },
        {
            "name": "Amaranth (rajgira)",
            "nutrients": ["iron", "calcium", "manganese"],
            "category": "grain",
        },
        {
            "name": "Sesame seeds (til)",
            "nutrients": ["calcium", "iron", "zinc"],
            "category": "seed",
        },
        {
            "name": "Turmeric milk (haldi doodh)",
            "nutrients": ["curcumin", "calcium", "vitamin_d"],
            "category": "beverage",
        },
        {
            "name": "Banana flower",
            "nutrients": ["iron", "fiber", "potassium"],
            "category": "vegetable",
        },
        {
            "name": "Ash gourd",
            "nutrients": ["vitamin_c", "fiber", "zinc"],
            "category": "vegetable",
        },
        {
            "name": "Hibiscus tea",
            "nutrients": ["anthocyanins", "vitamin_c", "iron"],
            "category": "beverage",
        },
    ],
}

_RISK_NUTRIENT_MAP: dict[str, list[str]] = {
    "cardiovascular": [
        "omega_3",
        "fiber",
        "magnesium",
        "potassium",
        "polyphenols",
        "CoQ10",
        "vitamin_K",
    ],
    "cancer": [
        "anthocyanins",
        "vitamin_c",
        "selenium",
        "curcumin",
        "sulforaphane",
        "ellagic_acid",
        "lycopene",
    ],
    "neurodegenerative": [
        "omega_3",
        "vitamin_d",
        "vitamin_e",
        "flavonoids",
        "B12",
        "folate",
        "lutein",
    ],
    "immune_dysfunction": [
        "vitamin_d",
        "zinc",
        "selenium",
        "probiotics",
        "vitamin_c",
        "beta_glucans",
        "glutathione_precursors",
    ],
    "premature_aging": ["folate", "B12", "vitamin_c", "zinc", "EGCG", "resveratrol", "astaxanthin"],
    "diabetes": [
        "chromium",
        "magnesium",
        "fiber",
        "alpha_lipoic_acid",
        "berberine",
        "omega_3",
        "vitamin_d",
    ],
    "obesity": [
        "fiber",
        "capsaicin",
        "EGCG",
        "chromium",
        "omega_3",
        "probiotics",
        "chlorogenic_acid",
    ],
    "alzheimers": ["omega_3", "vitamin_e", "folate", "B12", "curcumin", "flavonoids", "lutein"],
    "bone_health": ["calcium", "vitamin_d", "vitamin_K", "magnesium", "boron", "manganese", "zinc"],
    "blood_disorders": ["iron", "B12", "folate", "vitamin_c", "copper", "zinc", "vitamin_K"],
    "respiratory": [
        "vitamin_c",
        "vitamin_d",
        "omega_3",
        "quercetin",
        "magnesium",
        "selenium",
        "glutathione_precursors",
    ],
    "kidney": [
        "omega_3",
        "vitamin_d",
        "potassium",
        "magnesium",
        "probiotics",
        "alpha_lipoic_acid",
        "CoQ10",
    ],
    "liver": [
        "silymarin",
        "glutathione_precursors",
        "omega_3",
        "vitamin_e",
        "curcumin",
        "alpha_lipoic_acid",
        "selenium",
    ],
    "mental_health": ["omega_3", "B12", "folate", "vitamin_d", "magnesium", "zinc", "probiotics"],
    "reproductive": ["folate", "iron", "zinc", "omega_3", "vitamin_d", "CoQ10", "selenium"],
    "dermatological": [
        "vitamin_c",
        "vitamin_e",
        "omega_3",
        "zinc",
        "selenium",
        "beta_carotene",
        "probiotics",
    ],
    "gastrointestinal": [
        "probiotics",
        "fiber",
        "glutathione_precursors",
        "zinc",
        "omega_3",
        "quercetin",
        "vitamin_d",
    ],
    "endocrine": ["iodine", "selenium", "zinc", "vitamin_d", "omega_3", "magnesium", "iron"],
    "eye_health": [
        "lutein",
        "beta_carotene",
        "vitamin_c",
        "vitamin_e",
        "zinc",
        "omega_3",
        "astaxanthin",
    ],
    "dental": [
        "calcium",
        "vitamin_d",
        "vitamin_K",
        "manganese",
        "vitamin_c",
        "CoQ10",
        "probiotics",
    ],
    "sleep": ["magnesium", "B12", "vitamin_d", "L_theanine", "potassium", "calcium", "iron"],
    "hearing": ["folate", "omega_3", "magnesium", "vitamin_c", "vitamin_e", "zinc", "potassium"],
    "pain": ["omega_3", "curcumin", "gingerol", "magnesium", "vitamin_d", "capsaicin", "quercetin"],
    "infectious_disease": [
        "vitamin_c",
        "vitamin_d",
        "zinc",
        "selenium",
        "probiotics",
        "quercetin",
        "beta_glucans",
    ],
    "pharmacogenomics": [
        "vitamin_K",
        "folate",
        "B12",
        "omega_3",
        "magnesium",
        "CoQ10",
        "glutathione_precursors",
    ],
}


class NutritionAgent(BaseAgent):
    """Handles dietary recommendations based on genetic profile and geography.

    Listens for request messages with the following actions:

    * ``generate_diet_plan`` — build a personalised dietary plan.
    * ``get_protective_foods`` — list telomere-protective foods for a region.
    * ``adapt_to_preferences`` — adjust a plan for dietary restrictions.
    * ``calculate_gaps`` — identify nutritional gaps.

    Parameters
    ----------
    name : str
        Agent name (default ``"nutrition"``).
    """

    def __init__(self, name: str = "nutrition") -> None:
        super().__init__(
            name=name,
            capabilities=[
                "diet_planning",
                "nutritional_analysis",
                "regional_adaptation",
                "gap_analysis",
            ],
        )

    # ------------------------------------------------------------------
    # Message dispatch
    # ------------------------------------------------------------------

    async def handle_message(self, msg: AgentMessage) -> None:
        """Route an incoming request to the appropriate handler.

        Parameters
        ----------
        msg : AgentMessage
            Incoming message.
        """
        action = msg.content.get("action", "")
        handlers: dict[str, Any] = {
            "generate_diet_plan": self._handle_diet_plan,
            "get_protective_foods": self._handle_protective_foods,
            "adapt_to_preferences": self._handle_adapt,
            "calculate_gaps": self._handle_gaps,
        }

        handler = handlers.get(action)
        if handler is None:
            await self.send_message(
                recipient=msg.sender,
                content={"error": f"Unknown action '{action}'."},
                msg_type="error",
                correlation_id=msg.correlation_id,
            )
            return

        self.state = AgentState.RUNNING
        try:
            result = handler(msg.content)
            await self.send_message(
                recipient=msg.sender,
                content=result,
                msg_type="response",
                correlation_id=msg.correlation_id,
            )
        except Exception as exc:
            logger.exception("NutritionAgent action '%s' failed.", action)
            await self.send_message(
                recipient=msg.sender,
                content={"error": str(exc), "action": action},
                msg_type="error",
                correlation_id=msg.correlation_id,
            )
        finally:
            self.state = AgentState.IDLE

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_diet_plan(
        self,
        genetic_risks: list[dict[str, Any]],
        region: str = "global",
        profile: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Generate a personalised dietary plan based on genetic risks and geography.

        The plan prioritises nutrients that address the user's highest-risk
        conditions, sourced from foods available in the specified region.

        Parameters
        ----------
        genetic_risks : list[dict]
            Per-disease risk dicts (from :class:`GenomicsAgent`).
        region : str
            Geographic region for food sourcing (e.g. ``"mediterranean"``,
            ``"east_asian"``).  Falls back to ``"global"`` if unknown.
        profile : dict | None
            User metadata (``age``, ``sex``, ``dietary_restrictions``).

        Returns
        -------
        dict
            ``priority_nutrients``, ``recommended_foods``, ``meal_plan``,
            and ``notes``.
        """
        profile = profile or {}

        # Determine priority nutrients from top risks
        priority_nutrients = self._get_priority_nutrients(genetic_risks)

        # Get region-appropriate protective foods
        foods = self.get_telomere_protective_foods(region)

        # Score and rank foods by how many priority nutrients they supply
        scored_foods = self._score_foods(foods, priority_nutrients)

        # Build a simple daily meal plan
        meal_plan = self._build_meal_plan(scored_foods, profile)

        # Apply dietary restrictions
        restrictions = profile.get("dietary_restrictions", [])
        if restrictions:
            meal_plan = self._apply_restrictions(meal_plan, restrictions)

        return {
            "region": region,
            "priority_nutrients": priority_nutrients,
            "recommended_foods": scored_foods[:12],
            "meal_plan": meal_plan,
            "notes": self._generate_notes(genetic_risks, profile),
        }

    def get_telomere_protective_foods(
        self,
        region: str = "global",
    ) -> list[dict[str, Any]]:
        """Return telomere-protective foods available in the specified region.

        Parameters
        ----------
        region : str
            Geographic region identifier.

        Returns
        -------
        list[dict]
            Food entries with ``name``, ``nutrients``, and ``category``.
        """
        # Merge global foods with region-specific entries
        foods = list(_TELOMERE_PROTECTIVE_FOODS.get("global", []))
        regional = _TELOMERE_PROTECTIVE_FOODS.get(region, [])
        # Avoid duplicates by name
        existing_names = {f["name"] for f in foods}
        for item in regional:
            if item["name"] not in existing_names:
                foods.append(item)
                existing_names.add(item["name"])
        return foods

    def adapt_to_preferences(
        self,
        plan: dict[str, Any],
        restrictions: list[str],
    ) -> dict[str, Any]:
        """Adapt an existing diet plan to dietary restrictions.

        Supported restriction keywords: ``"vegetarian"``, ``"vegan"``,
        ``"gluten_free"``, ``"dairy_free"``, ``"nut_free"``,
        ``"shellfish_free"``.

        Parameters
        ----------
        plan : dict
            A diet plan as returned by :meth:`generate_diet_plan`.
        restrictions : list[str]
            List of restriction keywords.

        Returns
        -------
        dict
            Adjusted plan with unsuitable foods removed or substituted.
        """
        if not restrictions:
            return plan

        adapted = dict(plan)
        adapted["meal_plan"] = self._apply_restrictions(
            plan.get("meal_plan", {}),
            restrictions,
        )

        # Filter recommended foods
        adapted["recommended_foods"] = [
            f for f in plan.get("recommended_foods", []) if self._food_allowed(f, restrictions)
        ]

        adapted["dietary_restrictions_applied"] = restrictions
        return adapted

    def calculate_nutritional_gaps(
        self,
        current_diet: dict[str, Any],
        genetic_needs: list[str],
    ) -> dict[str, Any]:
        """Identify nutritional gaps between current diet and genetic needs.

        Parameters
        ----------
        current_diet : dict
            Mapping of nutrient names to approximate daily intake levels
            (e.g. ``{"omega_3": "low", "folate": "adequate"}``).
        genetic_needs : list[str]
            Priority nutrients derived from genetic risk analysis.

        Returns
        -------
        dict
            ``gaps`` (list of under-consumed nutrients), ``adequate``
            (list of met nutrients), ``recommendations`` (targeted
            suggestions).
        """
        gaps: list[str] = []
        adequate: list[str] = []

        for nutrient in genetic_needs:
            level = current_diet.get(nutrient, "unknown")
            if level in ("low", "deficient", "unknown"):
                gaps.append(nutrient)
            else:
                adequate.append(nutrient)

        recommendations: list[str] = []
        for gap in gaps:
            recommendations.append(
                f"Increase intake of {gap.replace('_', ' ')} through diet or supplementation."
            )

        return {
            "gaps": gaps,
            "adequate": adequate,
            "total_needs": len(genetic_needs),
            "gap_count": len(gaps),
            "gap_percentage": round(
                len(gaps) / max(len(genetic_needs), 1) * 100,
                1,
            ),
            "recommendations": recommendations,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_priority_nutrients(
        risks: list[dict[str, Any]],
    ) -> list[str]:
        """Determine priority nutrients from genetic risk scores."""
        nutrient_scores: dict[str, float] = {}
        for risk in risks:
            disease = risk.get("disease", "")
            score = risk.get("risk_score", 0.0)
            for nutrient in _RISK_NUTRIENT_MAP.get(disease, []):
                nutrient_scores[nutrient] = nutrient_scores.get(nutrient, 0.0) + score

        # Sort by cumulative relevance
        sorted_nutrients = sorted(
            nutrient_scores,
            key=nutrient_scores.get,  # type: ignore[arg-type]
            reverse=True,
        )
        return sorted_nutrients

    @staticmethod
    def _score_foods(
        foods: list[dict[str, Any]],
        priority_nutrients: list[str],
    ) -> list[dict[str, Any]]:
        """Score foods by how many priority nutrients they provide."""
        priority_set = set(priority_nutrients[:10])  # top 10 nutrients
        scored: list[dict[str, Any]] = []

        for food in foods:
            food_nutrients = set(food.get("nutrients", []))
            overlap = food_nutrients & priority_set
            scored.append(
                {
                    **food,
                    "relevance_score": len(overlap),
                    "matching_nutrients": sorted(overlap),
                }
            )

        scored.sort(key=lambda f: f["relevance_score"], reverse=True)
        return scored

    @staticmethod
    def _build_meal_plan(
        scored_foods: list[dict[str, Any]],
        profile: dict[str, Any],
    ) -> dict[str, Any]:
        """Build a simple daily meal plan from scored foods."""
        meals: dict[str, list[str]] = {
            "breakfast": [],
            "lunch": [],
            "dinner": [],
            "snacks": [],
        }

        category_meal_map: dict[str, str] = {
            "grain": "breakfast",
            "fruit": "breakfast",
            "beverage": "breakfast",
            "vegetable": "lunch",
            "legume": "lunch",
            "protein": "dinner",
            "fat": "dinner",
            "fermented": "lunch",
            "nuts": "snacks",
            "seed": "snacks",
            "spice": "dinner",
            "treat": "snacks",
            "dairy": "breakfast",
        }

        for food in scored_foods[:16]:
            category = food.get("category", "snacks")
            meal = category_meal_map.get(category, "snacks")
            if len(meals[meal]) < 4:
                meals[meal].append(food["name"])

        return {
            "daily_plan": meals,
            "servings_note": ("Aim for 5+ servings of fruits/vegetables daily."),
            "hydration": ("8+ glasses of water; consider green tea for EGCG."),
        }

    @staticmethod
    def _apply_restrictions(
        meal_plan: dict[str, Any],
        restrictions: list[str],
    ) -> dict[str, Any]:
        """Remove foods that violate dietary restrictions."""
        restriction_filters: dict[str, set[str]] = {
            "vegetarian": {
                "Barramundi",
                "Duck egg",
                "Egg white (cooked)",
                "Egg yolk (cooked)",
                "Eggs",
                "Elk meat",
                "Green-lipped mussel",
                "Grilled meat (nyama choma)",
                "Guinea fowl",
                "Herring",
                "Kangaroo meat",
                "Kippers (smoked herring)",
                "Lamb",
                "Lamb liver",
                "Mackerel",
                "Octopus",
                "Quail egg",
                "Salmon",
                "Salmon roe (ikura)",
                "Sardines",
                "Smoked mackerel",
                "Tofu",
                "Trout",
                "Turkey",
                "Wild salmon",
                "Wild-caught tuna",
            },
            "vegan": {
                "Aged cheddar cheese",
                "Barramundi",
                "Curd (yoghurt)",
                "Curd rice",
                "Dark chocolate (70%+)",
                "Duck egg",
                "Egg white (cooked)",
                "Egg yolk (cooked)",
                "Eggs",
                "Elk meat",
                "Fermented mare milk (kumis)",
                "Greek yoghurt",
                "Green-lipped mussel",
                "Grilled meat (nyama choma)",
                "Guinea fowl",
                "Herring",
                "Jaggery (gur)",
                "Kangaroo meat",
                "Kefir",
                "Kippers (smoked herring)",
                "Labneh",
                "Lamb",
                "Lamb liver",
                "Lavender honey",
                "Mackerel",
                "Manchego cheese",
                "Manuka honey",
                "Octopus",
                "Paneer",
                "Quail egg",
                "Quark",
                "Roquefort cheese",
                "Salmon",
                "Salmon roe (ikura)",
                "Sardines",
                "Smoked mackerel",
                "Tofu",
                "Trout",
                "Turkey",
                "Wild salmon",
                "Wild-caught tuna",
            },
            "gluten_free": set(),
            "dairy_free": {
                "Aged cheddar cheese",
                "Curd (yoghurt)",
                "Curd rice",
                "Fermented mare milk (kumis)",
                "Greek yoghurt",
                "Kefir",
                "Labneh",
                "Manchego cheese",
                "Paneer",
                "Quark",
                "Roquefort cheese",
            },
            "nut_free": {
                "Almonds",
                "Brazil nuts",
                "Chestnuts",
                "Groundnuts",
                "Groundnuts (peanuts)",
                "Macadamia nuts",
                "Marcona almonds",
                "Pecans",
                "Pine nuts",
                "Pistachios",
                "Tiger nuts",
                "Walnuts",
                "Walnuts (akhrot)",
            },
            "shellfish_free": set(),
        }

        excluded: set[str] = set()
        for restriction in restrictions:
            excluded |= restriction_filters.get(restriction, set())

        if not excluded:
            return meal_plan

        adapted = dict(meal_plan)
        daily = adapted.get("daily_plan", {})
        for meal_name, items in daily.items():
            if isinstance(items, list):
                daily[meal_name] = [i for i in items if i not in excluded]
        adapted["daily_plan"] = daily
        adapted["excluded_foods"] = sorted(excluded)
        return adapted

    @staticmethod
    def _food_allowed(
        food: dict[str, Any],
        restrictions: list[str],
    ) -> bool:
        """Check whether a food item is compatible with restrictions."""
        category_restrictions: dict[str, set[str]] = {
            "vegan": {"protein", "dairy", "treat"},
            "vegetarian": {"protein"},
            "dairy_free": {"dairy"},
            "nut_free": {"nuts"},
        }
        category = food.get("category", "")
        for restriction in restrictions:
            disallowed = category_restrictions.get(restriction, set())
            if category in disallowed:
                return False
        return True

    @staticmethod
    def _generate_notes(
        risks: list[dict[str, Any]],
        profile: dict[str, Any],
    ) -> list[str]:
        """Generate contextual notes for the diet plan."""
        notes: list[str] = [
            "This plan is a guideline — consult a registered dietitian for personalisation.",
        ]

        high_risks = [r for r in risks if r.get("risk_score", 0) >= 0.5]
        if high_risks:
            diseases = ", ".join(r["disease"].replace("_", " ") for r in high_risks)
            notes.append(
                f"Elevated risk detected for: {diseases}. Dietary focus adjusted accordingly."
            )

        age = profile.get("age", 0)
        if age > 60:
            notes.append("Over 60: consider B12 and vitamin D supplementation.")
        elif age < 25:
            notes.append("Under 25: focus on growth-supportive nutrients (iron, calcium, protein).")

        return notes

    # ------------------------------------------------------------------
    # Message handler bridges
    # ------------------------------------------------------------------

    def _handle_diet_plan(
        self,
        content: dict[str, Any],
    ) -> dict[str, Any]:
        return self.generate_diet_plan(
            genetic_risks=content.get("genetic_risks", []),
            region=content.get("region", "global"),
            profile=content.get("profile"),
        )

    def _handle_protective_foods(
        self,
        content: dict[str, Any],
    ) -> dict[str, Any]:
        foods = self.get_telomere_protective_foods(
            content.get("region", "global"),
        )
        return {
            "foods": foods,
            "region": content.get("region", "global"),
        }

    def _handle_adapt(
        self,
        content: dict[str, Any],
    ) -> dict[str, Any]:
        return self.adapt_to_preferences(
            plan=content.get("plan", {}),
            restrictions=content.get("restrictions", []),
        )

    def _handle_gaps(
        self,
        content: dict[str, Any],
    ) -> dict[str, Any]:
        return self.calculate_nutritional_gaps(
            current_diet=content.get("current_diet", {}),
            genetic_needs=content.get("genetic_needs", []),
        )
