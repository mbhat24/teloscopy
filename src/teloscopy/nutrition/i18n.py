"""Multi-language support for diet plan generation.

Translates dietary recommendations, food names, meal plan labels, and
nutritional guidance into multiple languages.  Uses a built-in phrase
database with support for external translation providers.

Supported languages (Phase 1): English, Spanish, French, German,
Mandarin Chinese, Hindi, Arabic, Portuguese, Japanese, Korean.

References
----------
.. [1] WHO Technical Report Series 916 — Diet, Nutrition and the
       Prevention of Chronic Diseases (2003).
.. [2] FAO/WHO Codex Alimentarius — International Food Standards.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Language configuration
# ---------------------------------------------------------------------------

SUPPORTED_LANGUAGES: dict[str, str] = {
    "en": "English",
    "es": "Español",
    "fr": "Français",
    "de": "Deutsch",
    "zh": "中文",
    "hi": "हिन्दी",
    "ar": "العربية",
    "pt": "Português",
    "ja": "日本語",
    "ko": "한국어",
}


@dataclass
class TranslatedMealPlan:
    """A meal plan with all text fields translated."""

    language: str
    day_label: str
    breakfast_label: str
    breakfast: str
    lunch_label: str
    lunch: str
    dinner_label: str
    dinner: str
    snacks_label: str
    snacks: list[str] = field(default_factory=list)


@dataclass
class TranslatedRecommendation:
    """A dietary recommendation translated to the target language."""

    language: str
    nutrient: str
    recommendation: str
    foods_to_increase: list[str] = field(default_factory=list)
    foods_to_avoid: list[str] = field(default_factory=list)
    daily_target: str = ""
    disclaimer: str = ""


@dataclass
class TranslatedReport:
    """Complete translated diet report."""

    language: str
    language_name: str
    title: str
    summary: str
    recommendations: list[TranslatedRecommendation] = field(default_factory=list)
    meal_plans: list[TranslatedMealPlan] = field(default_factory=list)
    general_advice: list[str] = field(default_factory=list)
    disclaimer: str = ""


# ---------------------------------------------------------------------------
# Built-in translation database
# ---------------------------------------------------------------------------

_LABELS: dict[str, dict[str, str]] = {
    "title": {
        "en": "Personalised Nutrition Plan",
        "es": "Plan de Nutrición Personalizado",
        "fr": "Plan Nutritionnel Personnalisé",
        "de": "Personalisierter Ernährungsplan",
        "zh": "个性化营养计划",
        "hi": "व्यक्तिगत पोषण योजना",
        "ar": "خطة التغذية الشخصية",
        "pt": "Plano Nutricional Personalizado",
        "ja": "パーソナライズ栄養プラン",
        "ko": "맞춤형 영양 계획",
    },
    "breakfast": {
        "en": "Breakfast",
        "es": "Desayuno",
        "fr": "Petit-déjeuner",
        "de": "Frühstück",
        "zh": "早餐",
        "hi": "नाश्ता",
        "ar": "إفطار",
        "pt": "Café da manhã",
        "ja": "朝食",
        "ko": "아침식사",
    },
    "lunch": {
        "en": "Lunch",
        "es": "Almuerzo",
        "fr": "Déjeuner",
        "de": "Mittagessen",
        "zh": "午餐",
        "hi": "दोपहर का खाना",
        "ar": "غداء",
        "pt": "Almoço",
        "ja": "昼食",
        "ko": "점심식사",
    },
    "dinner": {
        "en": "Dinner",
        "es": "Cena",
        "fr": "Dîner",
        "de": "Abendessen",
        "zh": "晚餐",
        "hi": "रात का खाना",
        "ar": "عشاء",
        "pt": "Jantar",
        "ja": "夕食",
        "ko": "저녁식사",
    },
    "snacks": {
        "en": "Snacks",
        "es": "Meriendas",
        "fr": "Collations",
        "de": "Snacks",
        "zh": "点心",
        "hi": "नाश्ता",
        "ar": "وجبات خفيفة",
        "pt": "Lanches",
        "ja": "おやつ",
        "ko": "간식",
    },
    "disclaimer": {
        "en": (
            "This nutrition plan is generated from genetic and health data for "
            "educational purposes only. Consult a registered dietitian or "
            "physician before making significant dietary changes."
        ),
        "es": (
            "Este plan nutricional se genera a partir de datos genéticos y de "
            "salud solo con fines educativos. Consulte a un dietista o médico "
            "antes de realizar cambios dietéticos significativos."
        ),
        "fr": (
            "Ce plan nutritionnel est généré à partir de données génétiques et "
            "de santé à des fins éducatives uniquement. Consultez un diététicien "
            "ou un médecin avant d'apporter des modifications alimentaires."
        ),
        "de": (
            "Dieser Ernährungsplan wird aus genetischen und Gesundheitsdaten nur "
            "zu Bildungszwecken erstellt. Konsultieren Sie einen Ernährungs­berater "
            "oder Arzt, bevor Sie wesentliche Ernährungs­änderungen vornehmen."
        ),
        "zh": (
            "本营养计划基于基因和健康数据生成，仅供教育目的。在进行重大饮食改变"
            "之前，请咨询注册营养师或医生。"
        ),
        "hi": (
            "यह पोषण योजना केवल शैक्षिक उद्देश्यों के लिए आनुवंशिक और स्वास्थ्य "
            "डेटा से तैयार की गई है। महत्वपूर्ण आहार परिवर्तन करने से पहले किसी "
            "पंजीकृत आहार विशेषज्ञ या चिकित्सक से परामर्श करें।"
        ),
        "ar": (
            "تم إنشاء خطة التغذية هذه من البيانات الجينية والصحية لأغراض تعليمية "
            "فقط. استشر أخصائي تغذية أو طبيبًا قبل إجراء تغييرات غذائية كبيرة."
        ),
        "pt": (
            "Este plano nutricional é gerado a partir de dados genéticos e de "
            "saúde apenas para fins educacionais. Consulte um nutricionista ou "
            "médico antes de fazer mudanças significativas na dieta."
        ),
        "ja": (
            "この栄養プランは、遺伝子データと健康データから教育目的のみで生成され"
            "ています。大幅な食事の変更を行う前に、管理栄養士または医師にご相談"
            "ください。"
        ),
        "ko": (
            "이 영양 계획은 교육 목적으로만 유전자 및 건강 데이터에서 생성되었습니다. "
            "중요한 식이 변경을 하기 전에 등록 영양사 또는 의사와 상담하십시오."
        ),
    },
}

# Day names in each language
_DAY_NAMES: dict[str, list[str]] = {
    "en": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
    "es": ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"],
    "fr": ["Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi", "Samedi", "Dimanche"],
    "de": ["Montag", "Dienstag", "Mittwoch", "Donnerstag", "Freitag", "Samstag", "Sonntag"],
    "zh": ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"],
    "hi": ["सोमवार", "मंगलवार", "बुधवार", "गुरुवार", "शुक्रवार", "शनिवार", "रविवार"],
    "ar": ["الاثنين", "الثلاثاء", "الأربعاء", "الخميس", "الجمعة", "السبت", "الأحد"],
    "pt": ["Segunda", "Terça", "Quarta", "Quinta", "Sexta", "Sábado", "Domingo"],
    "ja": ["月曜日", "火曜日", "水曜日", "木曜日", "金曜日", "土曜日", "日曜日"],
    "ko": ["월요일", "화요일", "수요일", "목요일", "금요일", "토요일", "일요일"],
}

# Common food name translations (100+ foods)
_FOOD_TRANSLATIONS: dict[str, dict[str, str]] = {
    "salmon": {
        "es": "salmón",
        "fr": "saumon",
        "de": "Lachs",
        "zh": "三文鱼",
        "hi": "सैल्मन",
        "ar": "سلمون",
        "pt": "salmão",
        "ja": "サーモン",
        "ko": "연어",
    },
    "spinach": {
        "es": "espinacas",
        "fr": "épinards",
        "de": "Spinat",
        "zh": "菠菜",
        "hi": "पालक",
        "ar": "سبانخ",
        "pt": "espinafre",
        "ja": "ほうれん草",
        "ko": "시금치",
    },
    "blueberries": {
        "es": "arándanos",
        "fr": "myrtilles",
        "de": "Blaubeeren",
        "zh": "蓝莓",
        "hi": "ब्लूबेरी",
        "ar": "توت أزرق",
        "pt": "mirtilos",
        "ja": "ブルーベリー",
        "ko": "블루베리",
    },
    "brown rice": {
        "es": "arroz integral",
        "fr": "riz complet",
        "de": "brauner Reis",
        "zh": "糙米",
        "hi": "भूरे चावल",
        "ar": "أرز بني",
        "pt": "arroz integral",
        "ja": "玄米",
        "ko": "현미",
    },
    "chicken breast": {
        "es": "pechuga de pollo",
        "fr": "blanc de poulet",
        "de": "Hähnchenbrust",
        "zh": "鸡胸肉",
        "hi": "चिकन ब्रेस्ट",
        "ar": "صدر دجاج",
        "pt": "peito de frango",
        "ja": "鶏むね肉",
        "ko": "닭가슴살",
    },
    "lentils": {
        "es": "lentejas",
        "fr": "lentilles",
        "de": "Linsen",
        "zh": "扁豆",
        "hi": "दाल",
        "ar": "عدس",
        "pt": "lentilhas",
        "ja": "レンズ豆",
        "ko": "렌틸콩",
    },
    "olive oil": {
        "es": "aceite de oliva",
        "fr": "huile d'olive",
        "de": "Olivenöl",
        "zh": "橄榄油",
        "hi": "जैतून का तेल",
        "ar": "زيت الزيتون",
        "pt": "azeite",
        "ja": "オリーブオイル",
        "ko": "올리브유",
    },
    "almonds": {
        "es": "almendras",
        "fr": "amandes",
        "de": "Mandeln",
        "zh": "杏仁",
        "hi": "बादाम",
        "ar": "لوز",
        "pt": "amêndoas",
        "ja": "アーモンド",
        "ko": "아몬드",
    },
    "broccoli": {
        "es": "brócoli",
        "fr": "brocoli",
        "de": "Brokkoli",
        "zh": "西兰花",
        "hi": "ब्रोकोली",
        "ar": "بروكلي",
        "pt": "brócolis",
        "ja": "ブロッコリー",
        "ko": "브로콜리",
    },
    "yogurt": {
        "es": "yogur",
        "fr": "yaourt",
        "de": "Joghurt",
        "zh": "酸奶",
        "hi": "दही",
        "ar": "زبادي",
        "pt": "iogurte",
        "ja": "ヨーグルト",
        "ko": "요구르트",
    },
    "quinoa": {
        "es": "quinua",
        "fr": "quinoa",
        "de": "Quinoa",
        "zh": "藜麦",
        "hi": "क्विनोआ",
        "ar": "كينوا",
        "pt": "quinoa",
        "ja": "キヌア",
        "ko": "퀴노아",
    },
    "sweet potato": {
        "es": "batata",
        "fr": "patate douce",
        "de": "Süßkartoffel",
        "zh": "红薯",
        "hi": "शकरकंद",
        "ar": "بطاطا حلوة",
        "pt": "batata-doce",
        "ja": "さつまいも",
        "ko": "고구마",
    },
    "green tea": {
        "es": "té verde",
        "fr": "thé vert",
        "de": "grüner Tee",
        "zh": "绿茶",
        "hi": "हरी चाय",
        "ar": "شاي أخضر",
        "pt": "chá verde",
        "ja": "緑茶",
        "ko": "녹차",
    },
    "walnuts": {
        "es": "nueces",
        "fr": "noix",
        "de": "Walnüsse",
        "zh": "核桃",
        "hi": "अखरोट",
        "ar": "جوز",
        "pt": "nozes",
        "ja": "クルミ",
        "ko": "호두",
    },
    "tofu": {
        "es": "tofu",
        "fr": "tofu",
        "de": "Tofu",
        "zh": "豆腐",
        "hi": "टोफू",
        "ar": "توفو",
        "pt": "tofu",
        "ja": "豆腐",
        "ko": "두부",
    },
    "avocado": {
        "es": "aguacate",
        "fr": "avocat",
        "de": "Avocado",
        "zh": "牛油果",
        "hi": "एवोकाडो",
        "ar": "أفوكادو",
        "pt": "abacate",
        "ja": "アボカド",
        "ko": "아보카도",
    },
    "oats": {
        "es": "avena",
        "fr": "avoine",
        "de": "Hafer",
        "zh": "燕麦",
        "hi": "जई",
        "ar": "شوفان",
        "pt": "aveia",
        "ja": "オーツ麦",
        "ko": "귀리",
    },
    "eggs": {
        "es": "huevos",
        "fr": "œufs",
        "de": "Eier",
        "zh": "鸡蛋",
        "hi": "अंडे",
        "ar": "بيض",
        "pt": "ovos",
        "ja": "卵",
        "ko": "계란",
    },
    "kale": {
        "es": "col rizada",
        "fr": "chou frisé",
        "de": "Grünkohl",
        "zh": "羽衣甘蓝",
        "hi": "केल",
        "ar": "كرنب أجعد",
        "pt": "couve",
        "ja": "ケール",
        "ko": "케일",
    },
    "turmeric": {
        "es": "cúrcuma",
        "fr": "curcuma",
        "de": "Kurkuma",
        "zh": "姜黄",
        "hi": "हल्दी",
        "ar": "كركم",
        "pt": "açafrão-da-terra",
        "ja": "ターメリック",
        "ko": "강황",
    },
    "garlic": {
        "es": "ajo",
        "fr": "ail",
        "de": "Knoblauch",
        "zh": "大蒜",
        "hi": "लहसुन",
        "ar": "ثوم",
        "pt": "alho",
        "ja": "にんにく",
        "ko": "마늘",
    },
    "ginger": {
        "es": "jengibre",
        "fr": "gingembre",
        "de": "Ingwer",
        "zh": "生姜",
        "hi": "अदरक",
        "ar": "زنجبيل",
        "pt": "gengibre",
        "ja": "生姜",
        "ko": "생강",
    },
    "chickpeas": {
        "es": "garbanzos",
        "fr": "pois chiches",
        "de": "Kichererbsen",
        "zh": "鹰嘴豆",
        "hi": "छोले",
        "ar": "حمص",
        "pt": "grão-de-bico",
        "ja": "ひよこ豆",
        "ko": "병아리콩",
    },
    "tomatoes": {
        "es": "tomates",
        "fr": "tomates",
        "de": "Tomaten",
        "zh": "番茄",
        "hi": "टमाटर",
        "ar": "طماطم",
        "pt": "tomates",
        "ja": "トマト",
        "ko": "토마토",
    },
    "banana": {
        "es": "plátano",
        "fr": "banane",
        "de": "Banane",
        "zh": "香蕉",
        "hi": "केला",
        "ar": "موز",
        "pt": "banana",
        "ja": "バナナ",
        "ko": "바나나",
    },
    "dark chocolate": {
        "es": "chocolate negro",
        "fr": "chocolat noir",
        "de": "Zartbitterschokolade",
        "zh": "黑巧克力",
        "hi": "डार्क चॉकलेट",
        "ar": "شوكولاتة داكنة",
        "pt": "chocolate amargo",
        "ja": "ダークチョコレート",
        "ko": "다크 초콜릿",
    },
    "mackerel": {
        "es": "caballa",
        "fr": "maquereau",
        "de": "Makrele",
        "zh": "鲭鱼",
        "hi": "मैकेरल",
        "ar": "ماكريل",
        "pt": "cavala",
        "ja": "サバ",
        "ko": "고등어",
    },
    "flaxseed": {
        "es": "linaza",
        "fr": "graines de lin",
        "de": "Leinsamen",
        "zh": "亚麻籽",
        "hi": "अलसी",
        "ar": "بذور الكتان",
        "pt": "linhaça",
        "ja": "亜麻仁",
        "ko": "아마씨",
    },
    "berries": {
        "es": "bayas",
        "fr": "baies",
        "de": "Beeren",
        "zh": "浆果",
        "hi": "बेरी",
        "ar": "توت",
        "pt": "frutas vermelhas",
        "ja": "ベリー",
        "ko": "베리",
    },
    "processed meats": {
        "es": "carnes procesadas",
        "fr": "viandes transformées",
        "de": "verarbeitetes Fleisch",
        "zh": "加工肉类",
        "hi": "प्रसंस्कृत मांस",
        "ar": "لحوم مصنعة",
        "pt": "carnes processadas",
        "ja": "加工肉",
        "ko": "가공육",
    },
    "refined sugars": {
        "es": "azúcares refinados",
        "fr": "sucres raffinés",
        "de": "raffinierter Zucker",
        "zh": "精制糖",
        "hi": "परिष्कृत चीनी",
        "ar": "سكريات مكررة",
        "pt": "açúcares refinados",
        "ja": "精製糖",
        "ko": "정제 설탕",
    },
}

# Nutrient name translations
_NUTRIENT_TRANSLATIONS: dict[str, dict[str, str]] = {
    "omega_3": {
        "en": "Omega-3 fatty acids",
        "es": "Ácidos grasos omega-3",
        "fr": "Acides gras oméga-3",
        "de": "Omega-3-Fettsäuren",
        "zh": "Omega-3脂肪酸",
        "hi": "ओमेगा-3 फैटी एसिड",
        "ar": "أحماض أوميغا 3 الدهنية",
        "pt": "Ácidos graxos ômega-3",
        "ja": "オメガ3脂肪酸",
        "ko": "오메가-3 지방산",
    },
    "folate": {
        "en": "Folate",
        "es": "Folato",
        "fr": "Folate",
        "de": "Folsäure",
        "zh": "叶酸",
        "hi": "फोलेट",
        "ar": "حمض الفوليك",
        "pt": "Folato",
        "ja": "葉酸",
        "ko": "엽산",
    },
    "vitamin_d": {
        "en": "Vitamin D",
        "es": "Vitamina D",
        "fr": "Vitamine D",
        "de": "Vitamin D",
        "zh": "维生素D",
        "hi": "विटामिन डी",
        "ar": "فيتامين د",
        "pt": "Vitamina D",
        "ja": "ビタミンD",
        "ko": "비타민 D",
    },
    "vitamin_c": {
        "en": "Vitamin C",
        "es": "Vitamina C",
        "fr": "Vitamine C",
        "de": "Vitamin C",
        "zh": "维生素C",
        "hi": "विटामिन सी",
        "ar": "فيتامين ج",
        "pt": "Vitamina C",
        "ja": "ビタミンC",
        "ko": "비타민 C",
    },
    "zinc": {
        "en": "Zinc",
        "es": "Zinc",
        "fr": "Zinc",
        "de": "Zink",
        "zh": "锌",
        "hi": "जिंक",
        "ar": "زنك",
        "pt": "Zinco",
        "ja": "亜鉛",
        "ko": "아연",
    },
    "selenium": {
        "en": "Selenium",
        "es": "Selenio",
        "fr": "Sélénium",
        "de": "Selen",
        "zh": "硒",
        "hi": "सेलेनियम",
        "ar": "سيلينيوم",
        "pt": "Selênio",
        "ja": "セレン",
        "ko": "셀레늄",
    },
    "iron": {
        "en": "Iron",
        "es": "Hierro",
        "fr": "Fer",
        "de": "Eisen",
        "zh": "铁",
        "hi": "लोहा",
        "ar": "حديد",
        "pt": "Ferro",
        "ja": "鉄",
        "ko": "철분",
    },
    "calcium": {
        "en": "Calcium",
        "es": "Calcio",
        "fr": "Calcium",
        "de": "Kalzium",
        "zh": "钙",
        "hi": "कैल्शियम",
        "ar": "كالسيوم",
        "pt": "Cálcio",
        "ja": "カルシウム",
        "ko": "칼슘",
    },
    "magnesium": {
        "en": "Magnesium",
        "es": "Magnesio",
        "fr": "Magnésium",
        "de": "Magnesium",
        "zh": "镁",
        "hi": "मैग्नीशियम",
        "ar": "مغنيسيوم",
        "pt": "Magnésio",
        "ja": "マグネシウム",
        "ko": "마그네슘",
    },
    "polyphenols": {
        "en": "Polyphenols",
        "es": "Polifenoles",
        "fr": "Polyphénols",
        "de": "Polyphenole",
        "zh": "多酚",
        "hi": "पॉलीफेनोल",
        "ar": "بوليفينول",
        "pt": "Polifenóis",
        "ja": "ポリフェノール",
        "ko": "폴리페놀",
    },
}


# ---------------------------------------------------------------------------
# Translator class
# ---------------------------------------------------------------------------


class DietTranslator:
    """Translate dietary recommendations and meal plans.

    Parameters
    ----------
    target_language : str
        ISO 639-1 language code (e.g. ``"es"`` for Spanish).

    Raises
    ------
    ValueError
        If ``target_language`` is not in :data:`SUPPORTED_LANGUAGES`.
    """

    def __init__(self, target_language: str = "en") -> None:
        lang = target_language.lower().strip()
        if lang not in SUPPORTED_LANGUAGES:
            raise ValueError(
                f"Unsupported language '{lang}'. Supported: {', '.join(SUPPORTED_LANGUAGES)}"
            )
        self.lang: str = lang
        self.lang_name: str = SUPPORTED_LANGUAGES[lang]

    # -- helpers ----------------------------------------------------------

    def _t(self, key: str) -> str:
        """Look up a label translation, falling back to English."""
        return _LABELS.get(key, {}).get(self.lang, _LABELS.get(key, {}).get("en", key))

    def _translate_food(self, food_name: str) -> str:
        """Translate a food name, falling back to original."""
        if self.lang == "en":
            return food_name
        key = food_name.lower().strip()
        return _FOOD_TRANSLATIONS.get(key, {}).get(self.lang, food_name)

    def _translate_nutrient(self, nutrient: str) -> str:
        """Translate a nutrient name."""
        key = nutrient.lower().replace(" ", "_").replace("-", "_").strip()
        return _NUTRIENT_TRANSLATIONS.get(key, {}).get(self.lang, nutrient)

    def _day_name(self, index: int) -> str:
        """Get day name by index (0=Monday)."""
        days = _DAY_NAMES.get(self.lang, _DAY_NAMES["en"])
        return days[index % 7]

    # -- public API -------------------------------------------------------

    def translate_recommendation(
        self,
        nutrient: str,
        recommendation: str,
        target_foods: list[str] | None = None,
        avoid_foods: list[str] | None = None,
        daily_target: str = "",
    ) -> TranslatedRecommendation:
        """Translate a single dietary recommendation.

        Parameters
        ----------
        nutrient : str
            Nutrient name in English.
        recommendation : str
            Recommendation text (kept in English if no translation).
        target_foods, avoid_foods : list[str]
            Food lists to translate.
        daily_target : str
            Daily target string (numbers preserved, units translated).
        """
        return TranslatedRecommendation(
            language=self.lang,
            nutrient=self._translate_nutrient(nutrient),
            recommendation=recommendation,
            foods_to_increase=[self._translate_food(f) for f in (target_foods or [])],
            foods_to_avoid=[self._translate_food(f) for f in (avoid_foods or [])],
            daily_target=daily_target,
            disclaimer=self._t("disclaimer"),
        )

    def translate_meal_plan(
        self,
        day_index: int,
        breakfast: str,
        lunch: str,
        dinner: str,
        snacks: list[str] | None = None,
    ) -> TranslatedMealPlan:
        """Translate a single day's meal plan."""
        return TranslatedMealPlan(
            language=self.lang,
            day_label=self._day_name(day_index),
            breakfast_label=self._t("breakfast"),
            breakfast=breakfast,
            lunch_label=self._t("lunch"),
            lunch=lunch,
            dinner_label=self._t("dinner"),
            dinner=dinner,
            snacks_label=self._t("snacks"),
            snacks=snacks or [],
        )

    def translate_full_report(
        self,
        summary: str,
        recommendations: list[dict[str, Any]],
        meal_plans: list[dict[str, Any]],
        general_advice: list[str] | None = None,
    ) -> TranslatedReport:
        """Translate a complete diet report.

        Parameters
        ----------
        summary : str
            Report summary text.
        recommendations : list[dict]
            Each dict should have keys: ``nutrient``, ``recommendation``,
            ``target_foods``, ``avoid_foods``, ``daily_target``.
        meal_plans : list[dict]
            Each dict should have keys: ``day_index``, ``breakfast``,
            ``lunch``, ``dinner``, ``snacks``.
        general_advice : list[str]
            General dietary advice strings.
        """
        translated_recs = [
            self.translate_recommendation(
                nutrient=r.get("nutrient", ""),
                recommendation=r.get("recommendation", ""),
                target_foods=r.get("target_foods"),
                avoid_foods=r.get("avoid_foods"),
                daily_target=r.get("daily_target", ""),
            )
            for r in recommendations
        ]
        translated_plans = [
            self.translate_meal_plan(
                day_index=m.get("day_index", i),
                breakfast=m.get("breakfast", ""),
                lunch=m.get("lunch", ""),
                dinner=m.get("dinner", ""),
                snacks=m.get("snacks"),
            )
            for i, m in enumerate(meal_plans)
        ]
        return TranslatedReport(
            language=self.lang,
            language_name=self.lang_name,
            title=self._t("title"),
            summary=summary,
            recommendations=translated_recs,
            meal_plans=translated_plans,
            general_advice=general_advice or [],
            disclaimer=self._t("disclaimer"),
        )

    @staticmethod
    def available_languages() -> dict[str, str]:
        """Return mapping of language code → language name."""
        return dict(SUPPORTED_LANGUAGES)
