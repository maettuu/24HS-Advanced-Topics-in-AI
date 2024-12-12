import difflib
import os

import torch
from fuzzywuzzy import process
from transformers import AutoModelForTokenClassification, AutoTokenizer

from app.config.app import settings
# Import base service with json entities (just much more simple stuff)
from app.services.extractors.main import Extractor, SpacyExtractor
from app.services.sparql_graph import SPARQLGraph


class MovieExtractor(Extractor):

    def __init__(self, spacy_extractor: SpacyExtractor = None):
        # self.bert_extractor = BERTMovieExtractor()
        # print("BERT extractor initialized")
        self.spacy_extractor = spacy_extractor if spacy_extractor else SpacyExtractor()
        self.score = 90  # for fuzzy match from NER
        super().__init__()

    def extract(self, text: str) -> list:
        # zero catch - call manual extract for simple cases
        # firstly try simple ntk corpus approach -> just from entities which we have rule based
        # secondly try embeddings and then try fuzzy search with trained model


        entities = self.get_nltk_entities(text)
        if entities:
            print("NLTK entity found:", entities)
            return entities

        print("No NLTK entity found, bert might help")

        # entities = self.bert_extraction(text)
        # if entities:
        #     print("Embedding entity found:", entities)
        #     return entities

        return []

    def get_nltk_entities(self, text: str) -> list:
        result = self.spacy_extractor.get_entities_with_fuzzy_matching(text, "movie")
        return result['movie']

    # def bert_extraction(self, text: str) -> list:

    #     print("NER extraction for ", text)
    #     result = self.bert_extractor.extract_entities(text)['entities']
    #     print("BERT extraction result:", result)
        
    #     for i, movie in enumerate(result):
    #         matched_movie = self.fuzzy_match_movie(movie)
    #         if matched_movie:
    #             result[i] = matched_movie

    #     return result

    # def fuzzy_match_movie(self, movie: str) -> str:

    #     # remove endings like " ,", [SEP] trailing spaces
    #     movie = movie.strip().strip(" ,").replace("[SEP]", "").strip()
    #     print("Fuzzy matching for movie:", movie)
    #     matches = self.spacy_extractor.get_fuzzy_matches_full_words(movie, 'movie', 80)
    #     # we need to filter out the match with the highest score
    #     # sort by score and take the first one
    #     matches = sorted(matches, key=lambda x: x['score'], reverse=True)
    #     print("Fuzzy matches:", matches)

        
        return matches[0] if matches else None


class BERTMovieExtractor:
    """Extract entities from the given text using a NER model."""

    def __init__(self, lazy_load: bool = False):
        self.model = None
        self.tokenizer = None
        self.lazy_load = lazy_load

        if not lazy_load:
            self._load_model()
        else:
            print("BERTMovieExtractor initialized in lazy load mode.")

    def _load_model(self):
        """Load the model and tokenizer from transformers."""
        model_name = os.sep.join(
            (settings.utils_path, "test_scripts", "custom_ner_on_q", "custom_ner_model"))  # Path to the custom model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(model_name)
        self.model.to(torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

    def extract_entities(self, text):
        """Extract entities from the given text."""
        if not self.model or not self.tokenizer:
            self._load_model()

        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, is_split_into_words=False)
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        inputs = {key: value.to(device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs).logits
        predictions = torch.argmax(outputs, dim=2)

        tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        label_list = ["O", "B-MOVIE", "I-MOVIE"]
        predicted_labels = [label_list[pred.item()] for pred in predictions[0]]

        entities = []
        for token, label in zip(tokens, predicted_labels):
            if label != "O":
                entities.append((token, label))

        return self.return_keywords(entities)

    def return_keywords(self, entities):
        """
        Return a dictionary with separate lists for entities and relations.
        Extracts multiple instances of entities and relations as separate items.
        """
        # Define keywords for easier expansion
        keyword_labels = {"MOVIE": ["B-MOVIE", "I-MOVIE"]}

        extracted_entities = {key: [] for key in keyword_labels}

        current_entity = []
        current_label = None

        for token, label in entities:
            for key, labels in keyword_labels.items():
                if label in labels:
                    if current_label and label.startswith("B-") and current_entity:
                        extracted_entities[current_label].append(" ".join(current_entity))
                        current_entity = []
                    current_entity.append(token)
                    current_label = key
                    break
            else:
                if current_entity:
                    extracted_entities[current_label].append(" ".join(current_entity))
                    current_entity = []
                    current_label = None

        # Append any remaining entities after the loop
        if current_entity:
            extracted_entities[current_label].append(" ".join(current_entity))

        return {"entities": extracted_entities["MOVIE"]}


# class FuzzyMatcher:
#     """Fuzzy match entities and relations  on the graph from the given text."""

#     def __init__(self, sparql_graph):
#         self.sparql_graph: SPARQLGraph = sparql_graph

#     def fuzzy_match_entity(self, entity: str) -> str:
#         graph_entities = self.sparql_graph.get_entities_labels()
#         best_entity_match = difflib.get_close_matches(entity, graph_entities, n=1, cutoff=0.6)

#         return best_entity_match[0] if best_entity_match else None

#     def fuzzy_match_relation(self, relationship) -> str:
#         graph_relations = self.sparql_graph.get_relations_labels()
#         best_relation_match = difflib.get_close_matches(relationship, graph_relations, n=1, cutoff=0.6)

#         return best_relation_match[0] if best_relation_match else None
