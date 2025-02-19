from enum import Enum
from typing import Optional


class ModelType(Enum):
    """Enum for model types and their corresponding endpoints."""

    CHAT = ("/v1/chat/completions", "chat")
    IMAGE = ("/v1/images/generations", "image")
    AUDIO = ("/v1/audio/speech", "audio")
    TRANSLATION = ("/v1/text/translations", "translation")
    EMBEDDINGS = ("/v1/embeddings", "embeddings")
    UPSCALE = ("/v1/images/upscale", "upscale")

    def __init__(self, endpoint: str, type_name: str) -> None:
        self.endpoint = endpoint
        self.type_name = type_name

    @classmethod
    def from_endpoint(cls, endpoint: str) -> Optional["ModelType"]:
        """Get ModelType from endpoint."""
        return next((mt for mt in cls if mt.endpoint == endpoint), None)


class TtsModals(Enum):
    ALLOY = "alloy"
    ECHO = "echo"
    FABLE = "fable"
    ONYX = "onyx"
    NOVA = "nova"
    SHIMMER = "shimmer"


class ElevenlabsModals(Enum):
    CHARLIE = "charlie"
    GEORGE = "george"
    CALLUM = "callum"
    LIAM = "liam"
    CHARLOTTE = "charlotte"
    ALICE = "alice"
    MATILDA = "matilda"
    CHRIS = "chris"
    BRIAN = "brian"
    DANIEL = "daniel"
    LILY = "lily"
    BILL = "bill"


class SpeechifyModals(Enum):
    HENRY = "henry"
    BWYNETH = "bwyneth"
    SNOOP = "snoop"
    MRBEAST = "mrbeast"
    GWYNETH = "gwyneth"
    CLIFF = "cliff"
    GUY = "guy"
    JANE = "jane"
    MATTHEW = "matthew"
    BENWILSON = "benwilson"
    PRESIDENTIAL = "presidential"
    CARLY = "carly"
    KYLE = "kyle"
    KRISTY = "kristy"
    OLIVER = "oliver"
    TASHA = "tasha"
    JOE = "joe"
    LISA = "lisa"
    GEORGE = "george"
    EMILY = "emily"
    ROB = "rob"
    RUSSELL = "russell"
    BENJAMIN = "benjamin"
    JENNY = "jenny"
    ARIA = "aria"
    JOANNA = "joanna"
    NATE = "nate"
    MARY = "mary"
    SALLI = "salli"
    JOEY = "joey"
    RYAN = "ryan"
    SONIA = "sonia"
    AMY = "amy"
    MICHAEL = "michael"
    THOMAS = "thomas"
    LIBBY = "libby"
    NARRATOR = "narrator"
    BRIAN = "brian"
    NATASHA = "natasha"
    WILLIAM = "william"
    FREYA = "freya"
    KEN = "ken"
    OLIVIA = "olivia"
    ADITI = "aditi"
    ABEO = "abeo"
    EZINNE = "ezinne"
    LUKE = "luke"
    LEAH = "leah"
    WILLEM = "willem"
    ADRI = "adri"
    FATIMA = "fatima"
    HAMDAN = "hamdan"
    HALA = "hala"
    RANA = "rana"
    BASSEL = "bassel"
    BASHKAR = "bashkar"
    TANISHAA = "tanishaa"
    KALINA = "kalina"
    BORISLAV = "borislav"
    JOANA = "joana"
    ENRIC = "enric"
    XIAOXIAO = "xiaoxiao"
    YUNFENG = "yunfeng"
    XIAOMENG = "xiaomeng"
    YUNJIAN = "yunjian"
    XIAOYAN = "xiaoyan"
    YUNZE = "yunze"
    ZHIYU = "zhiyu"
    HIUMAAN = "hiumaan"
    WANLUNG = "wanlung"
    HIUJIN = "hiujin"
    HSIAOCHEN = "hsiaochen"
    HSIAOYU = "hsiaoyu"
    YUNJHE = "yunjhe"
    SRECKO = "srecko"
    GABRIJELA = "gabrijela"
    ANTONIN = "antonin"
    VLASTA = "vlasta"
    CHRISTEL = "christel"
    JEPPE = "jeppe"
    COLETTE = "colette"
    MAARTEN = "maarten"
    LAURA = "laura"
    RUBEN = "ruben"
    DENA = "dena"
    ARNAUD = "arnaud"
    ANU = "anu"
    KERT = "kert"
    BLESSICA = "blessica"
    ANGELO = "angelo"
    HARRI = "harri"
    SELMA = "selma"
    DENISE = "denise"
    HENRI = "henri"
    CELESTE = "celeste"
    CLAUDE = "claude"
    SYLVIE = "sylvie"
    JEAN = "jean"
    CHARLINE = "charline"
    GERARD = "gerard"
    ARIANE = "ariane"
    FABRICE = "fabrice"
    KATJA = "katja"
    CHRISTOPH = "christoph"
    LOUISA = "louisa"
    CONRAD = "conrad"
    VICKI = "vicki"
    DANIEL = "daniel"
    GIORGI = "giorgi"
    EKA = "eka"
    ATHINA = "athina"
    NESTORAS = "nestoras"
    AVRI = "avri"
    HILA = "hila"
    MADHUR = "madhur"
    SWARA = "swara"
    NOEMI = "noemi"
    TAMAS = "tamas"
    GUDRUN = "gudrun"
    GUNNAR = "gunnar"
    GADIS = "gadis"
    ARDI = "ardi"
    IRMA = "irma"
    BENIGNO = "benigno"
    ELSA = "elsa"
    GIANNI = "gianni"
    PALMIRA = "palmira"
    DIEGO = "diego"
    IMELDA = "imelda"
    CATALDO = "cataldo"
    BIANCA = "bianca"
    ADRIANO = "adriano"
    MAYU = "mayu"
    NAOKI = "naoki"
    NANAMI = "nanami"
    DAICHI = "daichi"
    SHIORI = "shiori"
    KEITA = "keita"
    DAULET = "daulet"
    AIGUL = "aigul"
    SUNHI = "sunhi"
    INJOON = "injoon"
    JIMIN = "jimin"
    BONGJIN = "bongjin"
    SEOYEON = "seoyeon"
    ONA = "ona"
    LEONAS = "leonas"
    EVERITA = "everita"
    NILS = "nils"
    OSMAN = "osman"
    YASMIN = "yasmin"
    SAGAR = "sagar"
    HEMKALA = "hemkala"
    ISELIN = "iselin"
    FINN = "finn"
    PERNILLE = "pernille"
    FARID = "farid"
    DILARA = "dilara"
    AGNIESZKA = "agnieszka"
    MAREK = "marek"
    ZOFIA = "zofia"
    BRENDA = "brenda"
    DONATO = "donato"
    YARA = "yara"
    FABIO = "fabio"
    LEILA = "leila"
    JULIO = "julio"
    CAMILA = "camila"
    THIAGO = "thiago"
    FERNANDA = "fernanda"
    DUARTE = "duarte"
    INES = "ines"
    CRISTIANO = "cristiano"
    ALINA = "alina"
    EMIL = "emil"
    DARIYA = "dariya"
    DMITRY = "dmitry"
    TATYANA = "tatyana"
    MAXIM = "maxim"
    VIKTORIA = "viktoria"
    LUKAS = "lukas"
    PETRA = "petra"
    ROK = "rok"
    SAMEERA = "sameera"
    THILINI = "thilini"
    SAUL = "saul"
    VERA = "vera"
    ARNAU = "arnau"
    TRIANA = "triana"
    GERARDO = "gerardo"
    CARLOTA = "carlota"
    LUCIANO = "luciano"
    LARISSA = "larissa"
    LUPE = "lupe"
    HILLEVI = "hillevi"
    SOFIE = "sofie"
    REHEMA = "rehema"
    DAUDI = "daudi"
    PALLAVI = "pallavi"
    VALLUVAR = "valluvar"
    SARANYA = "saranya"
    KUMAR = "kumar"
    KANI = "kani"
    SURYA = "surya"
    VENBA = "venba"
    ANBU = "anbu"
    MOHAN = "mohan"
    SHRUTI = "shruti"
    PREMWADEE = "premwadee"
    NIWAT = "niwat"
    EMEL = "emel"
    AHMET = "ahmet"
    GUL = "gul"
    SALMAN = "salman"
    UZMA = "uzma"
    ASAD = "asad"
    POLINA = "polina"
    OSTAP = "ostap"
    HOAIMY = "hoaimy"
    NAMMINH = "namminh"
    ORLA = "orla"
    COLM = "colm"


class EmbeddingType(Enum):
    """Enum for supported embedding types."""

    FLOAT = "float"
    BASE64 = "base64"


class Language(Enum):
    """Enum for supported languages."""

    ABKHAZ = "ab"
    AFRIKAANS = "af"
    ALBANIAN = "sq"
    AMHARIC = "am"
    ARABIC = "ar"
    ARMENIAN = "hy"
    AZERBAIJANI = "az"
    BASQUE = "eu"
    BELARUSIAN = "be"
    BENGALI = "bn"
    BOSNIAN = "bs"
    BULGARIAN = "bg"
    BURMESE = "my"
    CATALAN = "ca"
    CEBUANO = "ceb"
    CHINESE = "zh"
    CROATIAN = "hr"
    CZECH = "cs"
    DANISH = "da"
    DUTCH = "nl"
    ENGLISH = "en"
    ESPERANTO = "eo"
    ESTONIAN = "et"
    FILIPINO = "fil"
    FINNISH = "fi"
    FRENCH = "fr"
    GALICIAN = "gl"
    GEORGIAN = "ka"
    GERMAN = "de"
    GREEK = "el"
    GUJARATI = "gu"
    HAITIAN_CREOLE = "ht"
    HAUSA = "ha"
    HEBREW = "he"
    HINDI = "hi"
    HMONG = "hmn"
    HUNGARIAN = "hu"
    ICELANDIC = "is"
    IGBO = "ig"
    INDONESIAN = "id"
    IRISH = "ga"
    ITALIAN = "it"
    JAPANESE = "ja"
    JAVANESE = "jv"
    KANNADA = "kn"
    KAZAKH = "kk"
    KHMER = "km"
    KOREAN = "ko"
    KURDISH = "ku"
    LAO = "lo"
    LATIN = "la"
    LATVIAN = "lv"
    LITHUANIAN = "lt"
    LUXEMBOURGISH = "lb"
    MACEDONIAN = "mk"
    MALAGASY = "mg"
    MALAY = "ms"
    MALAYALAM = "ml"
    MALTESE = "mt"
    MAORI = "mi"
    MARATHI = "mr"
    MONGOLIAN = "mn"
    NEPALI = "ne"
    NORWEGIAN = "no"
    PERSIAN = "fa"
    POLISH = "pl"
    PORTUGUESE = "pt"
    PUNJABI = "pa"
    ROMANIAN = "ro"
    RUSSIAN = "ru"
    SERBIAN = "sr"
    SINHALA = "si"
    SLOVAK = "sk"
    SLOVENIAN = "sl"
    SOMALI = "so"
    SPANISH = "es"
    SUNDANESE = "su"
    SWAHILI = "sw"
    SWEDISH = "sv"
    TAJIK = "tg"
    TAMIL = "ta"
    TELUGU = "te"
    THAI = "th"
    TURKISH = "tr"
    UKRAINIAN = "uk"
    URDU = "ur"
    UZBEK = "uz"
    VIETNAMESE = "vi"
    WELSH = "cy"
    XHOSA = "xh"
    YIDDISH = "yi"
    YORUBA = "yo"
    ZULU = "zu"
