from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch
import difflib
import spacy
from fuzzywuzzy import process
import json
from sentence_transformers import SentenceTransformer, util
import os
from torch.nn.functional import normalize

from app.services.extractors.main import Extractor, SpacyExtractor


class RelationshipExtractor(Extractor):

    def __init__(self, spacy_extractor: SpacyExtractor = None):
        if spacy_extractor:
            self.spacy_extractor = spacy_extractor
        else:
            self.spacy_extractor = SpacyExtractor()
            print("Spacy extractor initialized")
        self.embedding_extractor = RelationEmbeddingExtractor()
        print("Embedding extractor initialized")

    def manual_extract(self, text: str) -> dict:
        # if setence is like "When was the movie released?" then we can extract the relationship   
        # if on which day was the movie released or which year was the movie released

        if "released" in text.lower() and "when" in text.lower():
            return ["publication date"]

        if "released" in text.lower() and  ("year" in text.lower() or "date" in text.lower() or "day" in text.lower()):
            return ["publication date"]
        

        

    def extract(self, text: str) -> dict:
        # zero catch - call manual extract for simple cases
        # firstly try simple ntk corpus approach -> just from relationships which we have rule based
        # secondly try embeddings and then try fuzzy search with trained model

        # TODO add enum for security -> how sure are we about the statement, it can have an impact

        relations = self.manual_extract(text)
        if relations:
            print("Manual relationship found:", relations)
            return relations

        relations = self.get_nltk_relationships(text)
        if relations:
            print("NLTK relationship found:", relations)
            return relations
        
        relations = self.embedding_extraction(text)
        if relations:
            print("Embedding relationship found:", relations)
            return relations

        return []
        
    def get_nltk_relationships(self, text: str) -> list:
        result =  self.spacy_extractor.get_entities_with_fuzzy_matching(text, "relation")
        return result['relation']

    def embedding_extraction(self, text: str) -> list:
        # replace all weird characters like ?! etc
        to_replace = ["?", "!", ".", ",", ":", ";", "(", ")", "[", "]", "{", "}"]
        for char in to_replace:
            text = text.replace(char, "")

        print("Embedding extraction for ", text)
        # there is only one relation -> make list at the end
        result = self.embedding_extractor.find_most_similar_relation(text)

        return [result] if result else []
    
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

class RelationEmbeddingExtractor(Extractor):
    def __init__(self):
        super().__init__()
        # Placeholder for the transformer model and relation embeddings
        self.transformer_model = None
        self.relation_embeddings = None
        self.threshold = 0.5 # as is is just the last try? it might be enough

        self.load()
    
    def load(self, transformer_model_name='all-mpnet-base-v2'):
        """
        Loads relations and initializes the transformer model.
        :param relations_path: Path to the JSON file containing relations.
        :param transformer_model_name: Name of the SentenceTransformer model to use.
        """        
        # Load the transformer model
        self.transformer_model = SentenceTransformer(transformer_model_name)
        
        # Compute embeddings for all relations and store them
        self.relation_embeddings = normalize(self.transformer_model.encode(self.relations, convert_to_tensor=True))
    

    def preprocess_query(self, query):
        stop_words = set(stopwords.words("english"))
        tokens = word_tokenize(query.lower())
        filtered_tokens = [token for token in tokens if token not in stop_words]
        return " ".join(filtered_tokens)
    
    def find_most_similar_relation(self, query:str) -> str:
        """
        Finds the most similar relation to the given query based on cosine similarity.
        :param query: The user query as a string.
        :return: The most similar relation.
        """
        # remove stopwords and lowercase
        query = self.preprocess_query(query)

        query_embedding = self.transformer_model.encode(query, convert_to_tensor=True)
        # Encode the query to get the query embedding
        if query_embedding.dim() == 1:  # Reshape if it's 1D
            query_embedding = query_embedding.unsqueeze(0)

        # Normalize along the last dimension
        query_embedding = normalize(query_embedding, p=2, dim=1)
        
        # Compute cosine similarities between the query and all relation embeddings
        cosine_scores = util.pytorch_cos_sim(query_embedding, self.relation_embeddings) 

        # It doesnt matter where we flatten
        cosine_scores = cosine_scores.flatten()
        # Get the index of the most similar relation
        most_similar_idx = cosine_scores.argmax()

        
        # inspect the distance of the most similar relation
        most_similar_distance = cosine_scores[most_similar_idx]
        print(f"Most similar relation: {self.relations[most_similar_idx]} with distance: {most_similar_distance}")

        if most_similar_distance < self.threshold:
            print("Threshold not met")
            return None

        # Return the most similar relation
        print(f"Most similar relation: {self.relations[most_similar_idx]}")
        return self.relations[most_similar_idx]