"""Main pipeline class for breastfeeding NLP."""

import spacy
import medspacy
from medspacy.target_matcher import TargetMatcher
# from medspacy.context import ConTextComponent
from medspacy.context import ConText
from medspacy.section_detection import Sectionizer
# from medspacy.sentence_boundary_detection import PyRuSHComponent
from negspacy.negation import Negex
from negspacy.termsets import termset
from typing import Dict, Any, List, Optional, Union, Tuple, Set
from pydantic import BaseModel, Field
import uuid
from spacy.tokens import Doc, Span, Token

from breastfeeding_nlp.data.target_rules import get_target_rules, create_keyword_checker #, TargetRule
from breastfeeding_nlp.data.context_rules import get_context_rules #, ContextRule
from breastfeeding_nlp.data.section_rules import get_section_rules #, SectionRule
from breastfeeding_nlp.data.negation_terms import get_negation_termsets, get_chunk_prefixes
from breastfeeding_nlp.utils.preprocessing import preprocess_text, PreprocessingConfig
from breastfeeding_nlp.utils.export import to_huggingface_dataset, ExportConfig


class BreastfeedingNLPConfig(BaseModel):
    """Configuration for the breastfeeding NLP pipeline."""

    spacy_model: str = Field("en_core_web_sm", description="SpaCy model to use")
    use_medspacy_context: bool = Field(True, description="Use medspaCy's ConText algorithm")
    use_negspacy: bool = Field(True, description="Use negspaCy for negation detection")
    use_sectionizer: bool = Field(True, description="Use medspaCy's sectionizer")
    preprocessing_config: PreprocessingConfig = Field(
        default_factory=PreprocessingConfig, description="Configuration for text preprocessing"
    )
    export_config: ExportConfig = Field(
        default_factory=ExportConfig, description="Configuration for export to HuggingFace dataset"
    )


class BreastfeedingNLPPipeline:
    """
    Custom medspaCy pipeline for breastfeeding-related NER in clinical text.
    
    This pipeline integrates several medspaCy components:
    1. PyRuSH for sentence segmentation
    2. TargetMatcher for entity recognition
    3. ConText for contextual attribute assignment
    4. Sectionizer for section detection
    5. Negex for negation detection
    
    The pipeline also includes preprocessing steps and export functionality.
    """

    def __init__(self, 
                 nlp_method: str = "custom",
                 config: Optional[BreastfeedingNLPConfig] = None, 
                 target_rules_on_match: Optional[List[Dict[str, Any]]] = None,
                #  section_rules: Optional[List[Dict[str, Any]]] = None
                 ):
        """
        Initialize the pipeline with a configuration.
        
        Args:
            config: Configuration for the pipeline
        """
        if config is None:
            config = BreastfeedingNLPConfig()
        
        self.config = config
        self.initialize_nlp(nlp_method, config, target_rules_on_match)


    def initialize_nlp(self,
                       nlp_method: str = "custom",
                       config: Optional[BreastfeedingNLPConfig] = None,
                       target_rules_on_match: Optional[List[Dict[str, Any]]] = None):
        if nlp_method == "custom":
            self.initialize_nlp_custom(config, target_rules_on_match)
        elif nlp_method == "default":
            self.initialize_nlp_default(config)
        elif nlp_method == "mini":
            self.initialize_nlp_mini(config)

    def initialize_nlp_mini(self, config: BreastfeedingNLPConfig):
        """Just the sentence segmenter and target matcher"""
        # self.nlp = medspacy.load(self.config.spacy_model, enable=["medspacy_pyrush"])
        self.nlp = medspacy.load(self.config.spacy_model, disable=["ner"])
        self.nlp.add_pipe('sentencizer')
        from breastfeeding_nlp.data import target_rules_mini
        rules = target_rules_mini.get_target_rules()
        target_matcher = self.nlp.get_pipe("medspacy_target_matcher")
        target_matcher.add(rules)
        self.nlp.remove_pipe("medspacy_context")
        return self.nlp

    def initialize_nlp_default(self, config: BreastfeedingNLPConfig):
        self.nlp = medspacy.load(self.config.spacy_model)
        # add sectionizer
        self.nlp.add_pipe("medspacy_sectionizer", config={})
        # add negex
        self.nlp.add_pipe(
            "negex",
            config={
                "neg_termset": get_negation_termsets(),
                "chunk_prefix": get_chunk_prefixes(),
            },
        )
        # self._add_negation_dependency_check()
        rules = get_target_rules()
        target_matcher = self.nlp.get_pipe("medspacy_target_matcher")
        target_matcher.add(rules)

    def initialize_nlp_custom(self, config: BreastfeedingNLPConfig, target_rules_on_match: Optional[List[Dict[str, Any]]] = None):
        # Initialize spaCy with medspaCy
        self.nlp = medspacy.load(self.config.spacy_model, enable=["medspacy_pyrush"])
        # self.nlp = medspacy.load(self.config.spacy_model, disable=["ner", "medspacy_pyrush"]) #, enable=["medspacy_pyrush"])
        # print(self.nlp.pipe_names)
        #### NOTE: Parser and medspacy_pyrush are NOT compatible. Parser is general purpose and allows for dependency parsing.
        #### medspacy_pyrush is specialized for clinical text and does sentence segmentation.
        #### Need to empirically test both and see which is better for our purposes.

        # Add PyRuSH sentence segmentation with explicit default rules
        # self.nlp.add_pipe('sentencizer')#, before='parser') # can't split a sentence once it's been parsed.
        
        # Add section detection if enabled
        if config.use_sectionizer:
            self._add_sectionizer()
        
        # Add target matcher for entity recognition
        self._add_target_matcher(target_rules_on_match)
        
        # Add context if enabled
        if config.use_medspacy_context:
            self._add_context()
        
        # Add negspacy if enabled
        if config.use_negspacy:
            self._add_negspacy()
            # self._add_negation_dependency_check()
        
        # Add sentence classification component
        self._add_sentence_classifier()

        # Add intent classifier
        self._add_intent_classifier()

    def _add_sectionizer(self):
        """Add medspaCy sectionizer to the pipeline."""
        # Add sectionizer to the pipeline
        try:
            self.nlp.add_pipe("medspacy_sectionizer", before='parser')
        except ValueError: # already exists in the pipeline
            pass

        # Add the rules to the sectionizer
        # rules = get_section_rules()
        # sectionizer = self.nlp.get_pipe("medspacy_sectionizer")
        # sectionizer.add(rules)

    def _add_target_matcher(self, target_rules_on_match: Optional[List[Dict[str, Any]]] = None):
        """Add medspaCy TargetMatcher to the pipeline."""
        # Add TargetMatcher to the pipeline
        try:
            self.nlp.add_pipe("medspacy_target_matcher")
        except ValueError: # already exists in the pipeline
            pass

        # Add the rules to the target matcher -- These are the actual entities
        if target_rules_on_match is None:
            rules = get_target_rules()
        else:
            rules = get_target_rules(create_keyword_checker(target_rules_on_match))
        target_matcher = self.nlp.get_pipe("medspacy_target_matcher")
        target_matcher.add(rules)       

    def _add_context(self):
        """Add medspaCy ConText to the pipeline."""
        # Add ConText to the pipeline
        try:
            self.nlp.add_pipe("medspacy_context", last=True)
        except ValueError: # already exists in the pipeline
            pass

        # Add the rules to the target matcher
        rules = get_context_rules()
        context = self.nlp.get_pipe("medspacy_context")
        context.add(rules)
        # context_component = ConTextComponent(self.nlp, rules=rules)
        # self.nlp.add_pipe(context_component, last=True)
    
    def _add_intent_classifier(self):
        """Add custom component to classify sentences based on intent."""
        # Register the extension on the entity
        if not spacy.tokens.span.Span.has_extension("intent"):
            spacy.tokens.span.Span.set_extension("intent", default=False)
        
        # Define the component function
        @spacy.language.Language.component("intent_classifier")
        def classify_intent(doc):
            """Tag entities with intent flag if they are modified by an INTENT ConTextRule."""
            # Loop through entities
            for ent in doc.ents:
                # Check if this entity has any modifiers
                if hasattr(ent._, "modifiers") and ent._.modifiers:
                    # Check for INTENT modifiers
                    for modifier in ent._.modifiers:
                        if modifier.category == "INTENT":
                            # Set the intent flag to True
                            ent._.intent = True
                            break
            return doc
        
        # Add the component to the pipeline
        try:
            self.nlp.add_pipe("intent_classifier", after="medspacy_context")
        except ValueError:  # already exists in the pipeline
            pass

    def _add_negspacy(self):
        """Add negspaCy to the pipeline for negation detection."""
        # Get custom negation termsets
        # pseudo_negations, preceding_negations, following_negations, termination_terms = get_negation_termsets()
        # chunk_prefixes = get_chunk_prefixes()

        #######
        # NOTE: negspacy is not working as expected. Commenting out for now and adding in the preloaded en_clinical termset.
        #######
        # self.nlp.add_pipe(
        #     "negex",
        #     config={
        #         "neg_termset": get_negation_termsets(),
        #         "chunk_prefix": get_chunk_prefixes()
        #     },
        # )

        ts = termset("en_clinical")
        self.nlp.add_pipe("negex", config={"neg_termset": ts.get_patterns()})

    def check_negation(self, entity: Span) -> bool:
        """
        Return True if any negation cue in the sentence
        semantically governs 'entity', skipping blocks.
        """
        doc: Doc = entity.doc
        head: Token = entity.root
        # Precompute LCA matrix
        lca = doc.get_lca_matrix()

        termsets = get_negation_termsets()
        all_cues = set(
            # termsets["pseudo_negations"]
            termsets["preceding_negations"]
            + termsets["following_negations"]
            # + termsets["termination"]
        )

        BLOCK_NOUNS: Set[str] = {
            "difference",  # e.g. “denies a difference in X” → skip negating X
            "concern",
            "issue",
            "evidence",
            "problem",
            "absence",    # e.g. “denies an absence of X” → skip X
            "complaint",  # “denies complaints of X”
            "finding",    # “no findings of X”
            "sign",       # “no signs of X”
            "suggestion", # “no suggestion of X”
            "cause",      # “no cause of X”
            # —notice “history” is *not* here, so “no history of X” *will* negate X
        }

        for token in doc:
            if token.lemma_.lower() not in all_cues:
                continue
            i, j = token.i, head.i
            anc_idx = lca[i][j]
            if anc_idx < 0:
                continue
            ancestor = doc[anc_idx]

            # 1) If the LCA itself is a blocking noun, skip
            if ancestor.lemma_.lower() in BLOCK_NOUNS:
                continue

            # 2) If *any* blocker lies on the path from head → ancestor, skip
            node = head
            blocked = False
            while node != ancestor and node.head != node:
                if node.lemma_.lower() in BLOCK_NOUNS:
                    blocked = True
                    break
                node = node.head
            if blocked:
                continue

            # 3) Finally, ensure both cue and entity head are in that ancestor's subtree
            if token in ancestor.subtree and head in ancestor.subtree:
                return True

        return False

    def _add_negation_dependency_check(self):
        """Add custom component to check negation dependencies."""
        @spacy.language.Language.component("negation_scoper")
        def negation_scoper(doc: Doc) -> Doc:
            """
            Overrides ent._.negex based on our custom dependency check.
            Must come after the medSpaCy 'negex' component.
            """
            # run after medSpaCy’s negex so you have candidate cues
            # TODO: save custom negation check as a different attribute so both processes are saved.
            for ent in doc.ents:
                ent._.is_negated = self.check_negation(ent)
                # ent._.negex = self.check_negation(ent)
            return doc
        
        # Add the component to the pipeline
        if "negation_scoper" not in self.nlp.pipe_names:
            self.nlp.add_pipe("negation_scoper", after="negex")

    def _add_sentence_classifier(self):
        """Add custom component to classify sentences based on feeding entities."""
        # Register the extension on the span (for sentences)
        if not spacy.tokens.span.Span.has_extension("classification"):
            spacy.tokens.span.Span.set_extension("classification", default=None)
        
        # Define the component function
        @spacy.language.Language.component("sentence_classifier")
        def classify_sentences(doc):
            """
            Classify each sentence based on the entities it contains.
            NB: This is a simple classifier that does not apply any filters.
            """
            for sent in doc.sents:
                # Get all entities in this sentence
                entities = [ent for ent in doc.ents if sent.start <= ent.start < ent.end <= sent.end]
                
                # Check for breast feeding
                breast_feeding = any(ent.label_ == "BREAST_FEEDING" for ent in entities)
                
                # Check for formula feeding
                formula_feeding = any(ent.label_ == "FORMULA_FEEDING" for ent in entities)
                
                # Check for ambiguous feeding
                ambiguous = any(ent.label_ == "AMBIGUOUS" for ent in entities)
                
                # Check for feeding related but not definitive
                feeding_related = any(ent.label_ == "FEEDING_RELATED" for ent in entities)
                
                # Determine the classification
                if breast_feeding and formula_feeding:
                    classification = "MIXED"
                elif breast_feeding:
                    classification = "BREAST_FEEDING"
                elif formula_feeding:
                    classification = "FORMULA_FEEDING"
                elif ambiguous:
                    classification = "AMBIGUOUS"
                elif feeding_related:
                    classification = "NONE-FEEDING_RELATED"
                else:
                    classification = "NONE-NOT_FEEDING_RELATED"
                
                # Set the classification on the sentence
                sent._.set("classification", classification)
            
            return doc
        
        # Add the component to the pipeline
        if "sentence_classifier" not in self.nlp.pipe_names:
            self.nlp.add_pipe("sentence_classifier", last=True)

    def process(self, text: str, doc_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Process text through the full pipeline.
        
        Args:
            text: The input text to process
            doc_id: Optional document ID
            
        Returns:
            Dictionary with the processed document and metadata
        """
        # Preprocess text
        preprocess_result = preprocess_text(text, self.config.preprocessing_config)
        processed_text = preprocess_result["preprocessed_text"]
        
        # Process with medspaCy pipeline
        doc = self.nlp(processed_text)
        
        # Add document ID if provided
        if doc_id is None:
            doc_id = str(uuid.uuid4())
            
        # Register the doc_id extension if it doesn't exist
        if not spacy.tokens.Doc.has_extension("doc_id"):
            spacy.tokens.Doc.set_extension("doc_id", default=None)
        # Set the document ID
        doc._.doc_id = doc_id
        
        # Return the result with metadata
        result = {
            "doc": doc,
            "preprocessing": preprocess_result,
            "doc_id": doc_id,
        }
        
        return result

    def to_hf_ds(self, result: Dict[str, Any]) -> Any:
        """
        Convert processed results to a HuggingFace dataset.
        
        Args:
            result: Result from the process method
            
        Returns:
            HuggingFace dataset
        """
        doc = result["doc"]
        return to_huggingface_dataset(doc, self.config.export_config)

    def batch_process(self, texts: List[str], doc_ids: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Process a batch of texts.
        
        Args:
            texts: List of texts to process
            doc_ids: Optional list of document IDs
            
        Returns:
            List of processing results
        """
        if doc_ids is None:
            doc_ids = [str(uuid.uuid4()) for _ in range(len(texts))]
        
        results = []
        for text, doc_id in zip(texts, doc_ids):
            result = self.process(text, doc_id)
            results.append(result)
        
        return results 