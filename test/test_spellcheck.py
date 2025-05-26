# -*- coding: utf-8 -*-
from symspellpy import SymSpell, Verbosity
import re

sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
sym_spell.load_dictionary('french_dictionary.txt', term_index=0, count_index=1, separator=" ")

suggestions = sym_spell.lookup("orthgraphe", Verbosity.CLOSEST, max_edit_distance=2)
for suggestion in suggestions:
    print(suggestion.term)

input_text = f"""
Ceci est un texte avec des fautes d'orthgraphe.
Mouss Canard pome
Produit freis tranformation
Cuissson spécifique
Goupillion bleu avant philips
L'Oréal
Nike
Huawei
SAMSUNG
"""

words = re.findall(r"\b\w+\b|'", input_text.lower())
corrected_words = []

for word in words:
    if word == "'":
        corrected_words.append("'")
    else:
        suggestions = sym_spell.lookup(word, Verbosity.CLOSEST, max_edit_distance=2)
        if suggestions:
            corrected_words.append(suggestions[0].term)
        else:
            corrected_words.append(word)

final_words = []
i = 0
while i < len(corrected_words):
    if corrected_words[i] == "'" and i > 0 and i < len(corrected_words) - 1:
        combined = final_words.pop() + "'" + corrected_words[i + 1]
        final_words.append(combined)
        i += 2
    else:
        final_words.append(corrected_words[i])
        i += 1

corrected_text = " ".join(final_words)
print(corrected_text)
