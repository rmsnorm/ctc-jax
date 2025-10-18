"""Defines the phone set used in TIMIT."""

# fmt: off
_STOPS = ["b","d","g","p","t","k","dx"]
_STOP_CLOSURES = ["bcl","dcl","gcl","pcl","tck","kcl","tcl"]
_AFFRICATES = ["jh","ch"]
_FRICATIVES = ["s","sh","z","zh","f","th","v","dh"]
_NASALS = ["m","n","ng","em","en","eng","nx"]
_SEMIVOWEL_GLIDES = ["l","r","w","y","hh","hv","el"]
_VOWELS = ["iy","ih","eh","ey","ae","aa","aw","ay","ah","ao","oy","ow","uh",
           "uw","ux","er","ax","ix","axr","ax-h"]
# _OTHERS = ["pau","epi","h#","1","2"]
_OTHERS = ["h#", "pau", "epi"]
# fmt: on

PHONE_SET = (
    _STOPS
    + _STOP_CLOSURES
    + _AFFRICATES
    + _FRICATIVES
    + _NASALS
    + _SEMIVOWEL_GLIDES
    + _VOWELS
    + _OTHERS
)

# reduced symbols. symbols in the value are reduced to the key in the dict.
_REDUCED_SYMBOL_MAP = {
    "aa": ["aa", "ao"],
    "ah": ["ah", "ax", "ax-h"],
    "er": ["er", "axr"],
    "hh": ["hh", "hv"],
    "ih": ["ih", "ix"],
    "l": ["l", "el"],
    "m": ["m", "em"],
    "n": ["n", "en", "nx"],
    "ng": ["ng", "eng"],
    "sh": ["sh", "zh"],
    "sil": ["pcl", "tcl", "kcl", "bcl", "dcl", "gcl", "h#", "pau", "epi"],
}
_UNREDUCED_SYMBOLS = set(PHONE_SET) - set(
    [x for v in _REDUCED_SYMBOL_MAP.values() for x in v]
)
_UNREDUCED_SYMBOL_MAP = dict([(k, [k]) for k in _UNREDUCED_SYMBOLS])
FINAL_SYMBOL_MAP = _REDUCED_SYMBOL_MAP.copy()
FINAL_SYMBOL_MAP.update(_UNREDUCED_SYMBOL_MAP)
INVERTED_FINAL_SYMBOL_MAP = dict(
    [(x, k) for k, v in FINAL_SYMBOL_MAP.items() for x in v]
)
