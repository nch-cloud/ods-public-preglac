from itertools import combinations
from Levenshtein import ratio

def calculate_levenshtein_ratios(terms_list):
    res = {}
    for term1, term2 in combinations(terms_list, 2):
        levenshtein_ratio = ratio(term1, term2)
        res[f"{term1} vs {term2}"] = levenshtein_ratio
    return res

def deduplicate_similar_terms(terms_list, threshold=0.9):
    # Calculate ratios between all terms
    ratios = calculate_levenshtein_ratios(terms_list)

    if all(v == 1.0 for v in ratios.values()):
        # previous preprocessing has already deduplicated the terms
        return terms_list
    
    # Find pairs above threshold
    terms_to_remove = set()
    for pair, ratio_value in ratios.items():
        if ratio_value >= threshold:
            term1, term2 = pair.split(" vs ")
            # Before removing, check if this would impact other pairs negatively
            # For simplicity, we'll keep the first term and remove the second
            terms_to_remove.add(term2)
    
    # Return deduplicated list
    return [term for term in terms_list if term not in terms_to_remove]