"""Defines the phone set used in TIMIT."""

# fmt: off
_STOPS = ["b","d","g","p","t","k","dx","q"]
_STOP_CLOSURES = ["bcl","dcl","gcl","pcl","tck","kcl","tcl"]
_AFFRICATES = ["jh","ch"]
_FRICATIVES = ["s","sh","z","zh","f","th","v","dh"]
_NASALS = ["m","n","ng","em","en","eng","nx"]
_SEMIVOWEL_GLIDES = ["l","r","w","y","hh","hv","el"]
_VOWELS = ["iy","ih","eh","ey","ae","aa","aw","ay","ah","ao","oy","ow","uh",
           "uw","ux","er","ax","ix","axr","ax-h"]
_OTHERS = ["pau","epi","h#","1","2"]
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
