def rollup(term_name, ontology, root_id="FPO:0000001"):
    """
    Find the highest ancestor (excluding root node FPO:0000001) for a given term.
    
    Args:
        term_name (str): Name of the term to find ancestors for
        ontology (dict): Dictionary mapping term IDs to term data
        root_id (str): ID of the root node to stop at (default: "FPO:0000001")
        
    Returns:
        str: Name of the highest ancestor (excluding root)
    """
    # Find the term ID by name
    term_id = None
    for id, term_data in ontology.items():
        if term_data.get('name') == term_name:
            term_id = id
            break
    
    if not term_id:
        return None  # Term not found
    
    # Start from the current term and follow is_a relationships upward
    ancestors = []
    current_id = term_id

    while current_id and current_id != root_id:
        # Add current term to ancestors list
        if current_id != term_id:  # Skip the original term
            ancestors.append(current_id)
        
        # Get the parent term ID
        parent_id = None
        if 'is_a' in ontology[current_id]:
            parent_id = ontology[current_id]['is_a'].split(' ')[0]  # Extract just the ID part
        
        # Move to parent (or exit if no parent)
        current_id = parent_id
    
    # Return the highest ancestor (last in the list)
    if not ancestors:
        # return None  # No ancestors found (excluding root)
        ancestors = [term_id]
    
    highest_ancestor_id = ancestors[-1]
    return ontology[highest_ancestor_id]['name']

def load_ontology(file_path):
    """
    Load an ontology file in OBO-like format into a dictionary structure.
    
    Args:
        file_path (str): Path to the ontology file
        
    Returns:
        dict: Dictionary where keys are term IDs and values are dictionaries with term attributes
    """
    ontology = {}
    current_term = None
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        line = line.strip()
        
        # Skip empty lines and header lines
        if not line or line.startswith('format-version:') or line.startswith('data-version:') or line.startswith('ontology:'):
            continue
            
        # Start of a new term
        if line == '[Term]':
            current_term = {}
            continue
            
        # We're inside a term definition
        if current_term is not None and ':' in line:
            key, value = line.split(':', 1)
            value = value.strip()
            
            if key == 'id':
                term_id = value
                ontology[term_id] = current_term
            elif key in ['name', 'def', 'comment']:
                current_term[key] = value
            elif key == 'is_a':
                current_term[key] = value
            elif key == 'synonym':
                if 'synonyms' not in current_term:
                    current_term['synonyms'] = []
                # Extract just the synonym text part
                synonym_text = value.split('"')[1] if '"' in value else value
                current_term['synonyms'].append(synonym_text)
    
    return ontology

def main():
    """
    Example usage of the ontology functions.
    """
    # Load the ontology
    ontology = load_ontology("src/breastfeeding_nlp/data/ontology_prep/FPO.txt")
    
    # Test rollup with a term
    test_terms = ["Similac", "Breast feeding", "Bottlefeeding"]
    for term in test_terms:
        top_ancestor = rollup(term, ontology)
        print(f"Term: {term}, Top Ancestor: {top_ancestor}")
    
if __name__ == "__main__":
    main()