from app.services.disambiguation_service import DisambiguationService
from app.services.question_classifier import QuestionCategory
from app.services.sparql_graph import SPARQLGraph
from app.services.answering_service import AnsweringService
from app.services.extractors.main import SpacyExtractor

import random
import os
from app.config.app import settings
import json
import time


class ImageFinder(AnsweringService):

    # init with sparql graph
    def __init__(self, sparql_graph: SPARQLGraph, disambiguation: DisambiguationService, spacy_extractor: SpacyExtractor):
        self.sparql_graph: SPARQLGraph = sparql_graph
        self.disambiguation: DisambiguationService = disambiguation
        self.spacy_extractor: SpacyExtractor = spacy_extractor

        self.imdb_to_images = {}
        self.movie_to_poster = {}
        self._load_metadata()

    def _load_metadata(self):
        start_time = time.time()
        
        # load the json into a dictionary
        metadata_path = os.sep.join((settings.utils_path, "too_large_dataset", "movienet"))
        with open(os.path.join(metadata_path, 'structured_cast_images.json'), 'r') as file:
            self.imdb_to_images = json.load(file)

        metadata_path_2 = os.sep.join((settings.utils_path, "useful_dataset", "movienet"))
        with open(os.path.join(metadata_path_2, 'structured_movie_posters.json'), 'r') as file:
            self.movie_to_poster = json.load(file)

        end_time = time.time()
        print(f"Metadata loaded in {end_time - start_time} seconds")

    @staticmethod
    def _get_image(imdb_id: str, image_dict: dict) -> str | None:
        """Helper function to fetch a random image for a given IMDb ID from the provided dictionary."""
        if imdb_id not in image_dict.keys():
            print(f"No images found for IMDb ID: {imdb_id}")
            return None
        return random.choice(image_dict[imdb_id])

    def get_image_for_imdb_id(self, imdb_id: str) -> str:
        """Fetch a random image for a person IMDb ID."""
        return self._get_image(imdb_id, self.imdb_to_images)

    def get_image_for_movie_imdb_id(self, imdb_id: str) -> str:
        """Fetch a random poster for a movie IMDb ID."""
        return self._get_image(imdb_id, self.movie_to_poster)

    def _get_entity_imdb_id(self, entity: str) -> str:
        """Helper function to construct and execute a SPARQL query to fetch an IMDb ID for a given entity."""
        query = f"""
        PREFIX wd: <http://www.wikidata.org/entity/>
        PREFIX wdt: <http://www.wikidata.org/prop/direct/>
        SELECT ?imdb_id
        WHERE {{
            wd:{entity} wdt:P345 ?imdb_id .
        }}
        """

        response = self.sparql_graph.execute_query(query)

        return response if isinstance(response, str) else None

    def _get_entity_id(self, entity: str, entity_type: str):
        if entity_type == "person":
            return self.sparql_graph.get_uri_for_person(entity).split("/")[-1]
        return self.sparql_graph.get_uri_for_movie(entity).split("/")[-1]

    def _process_query(self, room_id: str, message: str, entity_type: str, image_function):
        """Generalized logic to process queries for both persons and movies."""
        entities = self.spacy_extractor.get_entities_with_fuzzy_matching(message, entity_type)
        print(f"Entities found: {entities}")

        if not entities.get(entity_type) or not entities[entity_type]:
            return None

        if len(entities[entity_type]) > 1:
            return None

        entity_string = entities[entity_type][0]
        print(f"{entity_type.capitalize()} string: {entity_string}")

        if entity_type == "person":
            if self.disambiguation.multiple_uris_for_entity(category=entity_type, label=entity_string):
                self.disambiguation.set_ambiguity(room_id, QuestionCategory.MULTIMEDIA, entity_type, entity_string)
                return self.disambiguation.make_human_response(room_id)
            entity_id = self._get_entity_id(entity_string, entity_type)
        else:  # "movie"
            entity_id = self._get_entity_id(entity_string, entity_type)

        if not entity_id:
            print(f"No URI found for {entity_type}.")
            return None

        return self.get_answer_and_make_response(entity_id, image_function, entity_string, entity_type)

    def get_answer_and_make_response(self, entity_id: str, image_function, entity_string: str, entity_type: str) -> str | None:
        imdb_id = self._get_entity_imdb_id(entity_id)

        if not imdb_id:
            return None

        random_image = image_function(imdb_id)
        if random_image is None:
            return None

        return self._format_answer(random_image, entity_string, entity_type)

    def get_response(self, room_id: str, message: str) -> str:
        person_response = self._process_query(room_id, message, "person", self.get_image_for_imdb_id)
        if person_response:
            return person_response

        movie_response = self._process_query(room_id, message, "movie", self.get_image_for_movie_imdb_id)
        if movie_response:
            return movie_response

        return "I'm sorry, but I can only show images of actors or movie posters. Please try rephrasing your query!"

    def get_image_for_movie(self, entity_uri: str) -> str | None:
        response = self._get_entity_imdb_id(entity_uri)
        if not response:
            return None

        random_image = self.get_image_for_movie_imdb_id(response)
        if not random_image:
            return None

        random_image = random_image.split(".")[0]
        return f"image:{random_image}"

    @staticmethod
    def _format_answer(random_image_str: str, entity_string: str, entity_type: str):
        random_image_str = random_image_str.split(".")[0]

        templates = {
            "person": [
                f"image:{random_image_str} There is an image of {entity_string}, enjoy!",
                f"image:{random_image_str} Here is an image of {entity_string} for you!",
                f"image:{random_image_str} This is a picture of {entity_string}.",
                f"image:{random_image_str} Enjoy this image of {entity_string}.",
                f"image:{random_image_str} Here is a picture of {entity_string}.",
                f"image:{random_image_str} This is a photo of {entity_string}.",
                f"image:{random_image_str} Take a look at this wonderful image of {entity_string}!",
                f"image:{random_image_str} Feast your eyes on this picture of {entity_string}.",
                f"image:{random_image_str} Here's a fantastic shot of {entity_string}, just for you!",
                f"image:{random_image_str} Behold this captivating image of {entity_string}!",
                f"image:{random_image_str} Hope you enjoy this striking photo of {entity_string}.",
                f"image:{random_image_str} Check out this amazing picture of {entity_string}.",
                f"image:{random_image_str} Here's a stunning photo of {entity_string} to enjoy!",
                f"image:{random_image_str} Here's an impressive snapshot of {entity_string}.",
                f"image:{random_image_str} Take a moment to appreciate this photo of {entity_string}.",
                f"image:{random_image_str} Look at this beautiful picture of {entity_string}!"
            ],
            "movie": [
                f"image:{random_image_str} There is a poster of the movie {entity_string}, enjoy!",
                f"image:{random_image_str} Here is a poster of the movie {entity_string} for you!",
                f"image:{random_image_str} This is a poster of the movie {entity_string}.",
                f"image:{random_image_str} Enjoy this poster of the movie {entity_string}.",
                f"image:{random_image_str} Here is a poster of the movie {entity_string}.",
                f"image:{random_image_str} This is a poster of the movie {entity_string}.",
                f"image:{random_image_str} Take a look at this wonderful poster of the movie {entity_string}!",
                f"image:{random_image_str} Feast your eyes on this poster of the movie {entity_string}.",
                f"image:{random_image_str} Here's a fantastic poster of the movie {entity_string}, just for you!",
                f"image:{random_image_str} Behold this captivating poster of the movie {entity_string}!",
                f"image:{random_image_str} Hope you enjoy this striking poster of the movie {entity_string}.",
                f"image:{random_image_str} Check out this amazing poster of the movie {entity_string}.",
                f"image:{random_image_str} Here's a stunning poster of the movie {entity_string} to enjoy!",
                f"image:{random_image_str} Here's an impressive poster of the movie {entity_string}.",
                f"image:{random_image_str} Take a moment to appreciate this poster of the movie {entity_string}.",
                f"image:{random_image_str} Look at this beautiful poster of the movie {entity_string}!"
            ]
        }

        return random.choice(templates[entity_type])
