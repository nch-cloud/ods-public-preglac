import medspacy
from breastfeeding_nlp.data.target_rules import get_target_rules

class BreasteedingNLPPipelineMini:
    """
    A mini version of the BreastfeedingNLPPipeline that only does NER and has no context awareness.
    """
    def __init__(self):
        self.nlp = _initialize_pipeline()
    
    def _initialize_pipeline(self):
        self.nlp = medspacy.load()
        self.nlp.add_pipe("medspacy_sectionizer", config={})
        rules = get_target_rules()
        target_matcher = self.nlp.get_pipe("medspacy_target_matcher")
        target_matcher.add(rules)