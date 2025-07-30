import numpy as np
from typing import Any, Dict, List
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.metrics.pairwise import cosine_similarity

# from breastfeeding_nlp.llm.agents import BedrockClient
from breastfeeding_nlp.pipeline import BreastfeedingNLPPipeline, BreastfeedingNLPConfig
from breastfeeding_nlp.utils.preprocessing import PreprocessingConfig
from breastfeeding_nlp.llm.query import query_llm

def create_section_chunks(text: str, doc_id: str, chunk_size: int = 500) -> Dict[str, Dict[str, Any]]:
    """Create chunks from spaCy Doc sections"""
    spacy_doc = sectionize_text(text, doc_id)
    section_chunks = []
    for section in spacy_doc._.sections:
        section_info = section.serialized_representation()
        # Get the section text including title and body
        section_text = spacy_doc.text[section_info['title_start']: section_info['body_end']]
        
        # Extend to sentence boundaries
        # Find the beginning of the first complete sentence
        sentence_start = section_info['title_start']
        if sentence_start > 0:
            # Look for previous sentence boundary
            prev_boundaries = [spacy_doc.text.rfind(p, max(0, sentence_start-50), sentence_start) 
                              for p in ['. ', '? ', '! ', '.\n', '?\n', '!\n', '\n\n']]
            prev_boundary = max([b for b in prev_boundaries if b != -1], default=-1)
            if prev_boundary != -1:
                sentence_start = prev_boundary + 2  # +2 to skip the punctuation and space
        
        # Find the end of the last complete sentence
        sentence_end = section_info['body_end']
        if sentence_end < len(spacy_doc.text):
            # Look for next sentence boundary
            next_boundaries = [spacy_doc.text.find(p, sentence_end, min(len(spacy_doc.text), sentence_end+50)) 
                              for p in ['. ', '? ', '! ', '.\n', '?\n', '!\n', '\n\n']]
            next_boundary = min([b for b in next_boundaries if b != -1], default=-1)
            if next_boundary != -1:
                sentence_end = next_boundary + 1  # +1 to include the punctuation
        
        # Get the extended section text
        section_text = spacy_doc.text[sentence_start:sentence_end]
        
        # Split section text into chunks of specified size
        if len(section_text) <= chunk_size:
            section_chunks.append(section_text)
        else:
            # Process the text in chunks
            start = 0
            while start < len(section_text):
                end = min(start + chunk_size, len(section_text))
                
                # If we're not at the end of the text and not at a sentence boundary,
                # extend to the next sentence boundary
                if end < len(section_text):
                    # Look for sentence ending punctuation followed by space or newline
                    next_sentence_end = -1
                    for punct in ['. ', '? ', '! ', '.\n', '?\n', '!\n', '\n']:
                        pos = section_text.find(punct, end - 10, end + 50)
                        if pos != -1 and (next_sentence_end == -1 or pos < next_sentence_end):
                            next_sentence_end = pos + len(punct) - 1
                    
                    if next_sentence_end != -1:
                        end = next_sentence_end + 1
                
                section_chunks.append(section_text[start:end])
                start = end
    
    return section_chunks

def sectionize_text(text: str, doc_id: str) -> Dict[str, Dict[str, Any]]:
    preproc_config = PreprocessingConfig(pre_expand_abbreviations=False, typo_resolution=False)
    main_config = BreastfeedingNLPConfig(preprocessing_config=preproc_config)
    pipeline = BreastfeedingNLPPipeline(
        nlp_method='custom',
        config=main_config,
    )

    return pipeline.process(text=text, doc_id=doc_id)['doc']

class HydeRagPipeline:
    def __init__(self):
        print("Initializing HyDE-RAG pipeline with reranking...")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2') # TODO: Consider ModernBERT instead.
        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        self.chunks = []
        self.chunk_embeddings = None
        
    def index_documents(self, documents: List[str], doc_id: str):
        """Process and index documents for retrieval"""
        self.chunks = create_section_chunks(documents, doc_id)
        print(f"Created {len(self.chunks)} chunks from the clinical notes")
        
        # Generate embeddings for all chunks
        print("Generating embeddings for document chunks...")
        self.chunk_embeddings = self.embedding_model.encode(self.chunks, 
                                                           show_progress_bar=True,
                                                           convert_to_tensor=True)
    
    def generate_hyde_document(self, query: str) -> str:
        """
        Generate a hypothetical document that answers the query using an LLM.
        """
        print(f"Generating hypothetical document for query: '{query}'")
        
        prompt = f"""
        You are a clinical documentation expert. Given a clinical query, generate a hypothetical 
        document that would contain the answer to this query. The document should mimic the style 
        and structure of a typical clinical note or medical record.
        
        Query: {query}
        
        Hypothetical Document:
        """
        # TODO: adjust the prompt to be more specific to the query and more concise. Shorter response and more structured.
        # TODO: Look into structured output with Pydantic or from anthropic/openai.
        # TODO: Haiku might be good for this.
        
        return query_llm(system_prompt=None, text=prompt)['text'] # add haiku

        # # In a real implementation, this would call an LLM API with the prompt
        # # For this example, we're simulating the response
        
        # # Simple pattern matching to generate hypothetical documents for clinical queries
        # if "diabetes" in query.lower() and "medication" in query.lower():
        #     return """
        #     In a typical clinical note, diabetes medications would be listed in the medications section.
        #     Common medications for diabetes include metformin, which is often first-line therapy.
        #     Some patients may be prescribed sulfonylureas like glipizide or glyburide.
        #     For patients with more advanced diabetes, insulin therapy might be documented,
        #     including long-acting insulins like glargine and rapid-acting insulins like lispro.
        #     The dosages and frequencies would typically be noted, such as 'Metformin 1000mg BID'
        #     or 'Insulin glargine 30 units at bedtime'.
        #     """
        # elif "lab" in query.lower() and "diabetes" in query.lower():
        #     return """
        #     For patients with diabetes, laboratory values typically include HbA1c percentages,
        #     which reflect average blood glucose over the past 3 months. Target HbA1c is generally
        #     below 7% for most patients. Fasting glucose values would also be reported, typically
        #     in mg/dL, with normal fasting glucose below 100 mg/dL. Elevated values would be noted
        #     and might trigger medication adjustments.
        #     """
        # else:
        #     # Generic hypothetical document for other clinical queries
        #     return """
        #     Clinical notes typically contain structured sections including patient demographics,
        #     medical history, medications list, vital signs, laboratory results, and assessment/plan.
        #     For specific conditions like diabetes, hypertension, or COPD, the notes would include
        #     current medications, relevant lab values, and treatment plans for these conditions.
        #     Medications would be listed with dosages and frequencies. Laboratory values would be
        #     provided with reference ranges or indications if they are abnormal.
        #     """
    
    def retrieve_with_hyde(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieve relevant chunks using Hypothetical Document Embeddings (HyDE)
        """
        # Generate hypothetical document
        self.hyde_doc = self.generate_hyde_document(query)
        print(f"Generated hypothetical document (excerpt): {self.hyde_doc[:100]}...")
        
        # Create embedding for the hypothetical document
        hyde_embedding = self.embedding_model.encode(self.hyde_doc, convert_to_tensor=True)
        
        # Compute similarities
        similarities = cosine_similarity(
            hyde_embedding.cpu().numpy().reshape(1, -1),
            self.chunk_embeddings.cpu().numpy()
        )[0]
        
        # Get top_k most similar chunks
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = []
        for i, idx in enumerate(top_indices):
            results.append({
                'chunk': self.chunks[idx],
                'similarity': float(similarities[idx]),
                'rank': i + 1
            })
            
        return results
    
    def rerank_results(self, query: str, initial_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Rerank the retrieved chunks using a cross-encoder
        """
        print("Reranking initial results...")
        
        # Prepare pairs for the cross-encoder
        pairs = [(query, result['chunk']) for result in initial_results]
        
        # Get reranking scores
        rerank_scores = self.reranker.predict(pairs)
        
        # Add reranking scores to results
        for i, score in enumerate(rerank_scores):
            initial_results[i]['rerank_score'] = float(score)
        
        # Sort by reranking score
        reranked_results = sorted(initial_results, key=lambda x: x['rerank_score'], reverse=True)
        
        # Update ranks
        for i, result in enumerate(reranked_results):
            result['rerank'] = i + 1
            
        return reranked_results
    
    def generate_answer(self, query: str, reranked_results: List[Dict[str, Any]], top_k: int = 3) -> str:
        """
        Generate an answer based on the reranked results using an LLM.
        """
        print(f"Generating answer using top {top_k} reranked chunks...")
        
        top_chunks = [result['chunk'] for result in reranked_results[:top_k]]
        context = "\n\n".join(top_chunks)
        
        prompt = f"""
        You are a clinical assistant helping healthcare professionals find information in medical records.
        
        QUERY: {query}
        
        RELEVANT CLINICAL NOTES:
        {context}
        
        Based on the provided clinical notes, please answer the query. Focus only on information that is 
        explicitly mentioned in the notes. If the information is not available in the provided context, 
        state that clearly. Format your response with bullet points for clarity when appropriate.
        
        ANSWER:
        """
        
        response = query_llm(system_prompt=None, text=prompt)['text']

        # # In a real implementation, this would call an LLM API with the prompt
        # # For this example, we're simulating the response
        
        # # Simulate an LLM response based on the retrieved context
        # response = f"Based on analysis of the clinical notes, here's what I found regarding: '{query}'\n\n"
        
        # if "diabetes" in query.lower() and "medication" in query.lower():
        #     if "Metformin" in context:
        #         response += "• Metformin: Found in multiple patient records, typically dosed at 500-1500mg BID for diabetes management\n"
        #     if "Insulin" in context:
        #         response += "• Insulin therapy: Some patients are on both long-acting (glargine) and rapid-acting (lispro) insulins\n"
        #     if "Glipizide" in context:
        #         response += "• Glipizide: 5mg daily noted for at least one patient, typically for type 2 diabetes\n"
            
        #     response += "\nRecent medication changes noted:\n"
        #     if "Increase Metformin" in context:
        #         response += "• Increased Metformin dosage due to suboptimal diabetes control\n"
                
        # elif "lab" in query.lower():
        #     if "HbA1c" in context:
        #         response += "HbA1c values across patients range from 7.1% to 8.2%, with most above target range\n"
        #     if "Glucose" in context:
        #         response += "Fasting glucose values range from 152-185 mg/dL, indicating elevated levels\n"
                
        return response
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """
        Process a query through the complete HyDE-RAG pipeline with reranking
        """
        # 1. Initial retrieval with HyDE
        print("\n=== STEP 1: Initial Retrieval with HyDE ===")
        initial_results = self.retrieve_with_hyde(query, top_k=10)
        
        # 2. Rerank results
        print("\n=== STEP 2: Reranking Results ===")
        reranked_results = self.rerank_results(query, initial_results)
        
        # 3. Generate answer
        print("\n=== STEP 3: Generating Final Answer ===")
        answer = self.generate_answer(query, reranked_results, top_k=3)
        
        return {
            "query": query,
            "initial_results": initial_results,
            "reranked_results": reranked_results,
            "answer": answer
        }

def display_results(results):
    """Format and display the results of the pipeline"""
    print("\n=== Initial Retrieval Results (HyDE) ===")
    print(f"Top 5 results from {len(results['initial_results'])} retrieved chunks:")
    for i, result in enumerate(results['initial_results'][:5]):
        print(f"{i+1}. Similarity: {result['similarity']:.4f}")
        print(f"   Chunk: {result['chunk'][:150].replace(chr(10), ' ')}...")
    
    print("\n=== Reranked Results ===")
    print("Top 5 results after reranking:")
    for i, result in enumerate(results['reranked_results'][:5]):
        print(f"{i+1}. Rerank Score: {result['rerank_score']:.4f} (was rank {result['rank']})")
        print(f"   Chunk: {result['chunk'][:150].replace(chr(10), ' ')}...")
    
    print("\n=== Final Answer ===")
    print(results['answer'])

