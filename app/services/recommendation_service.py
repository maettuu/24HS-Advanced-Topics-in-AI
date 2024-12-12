import random

import rdflib

from app.config.app import settings
from app.services.answering_service import AnsweringService
from app.services.disambiguation_service import DisambiguationService
from app.services.extractors.main import SpacyExtractor
from app.services.extractors.movie_extraction import MovieExtractor
from app.services.llm_service import LlmService
from app.services.question_classifier import QuestionCategory
from app.services.image_finder import ImageFinder
from app.services.sparql_graph import SPARQLGraph


class RecommendationService(AnsweringService):
    def __init__(self, sparql_graph: SPARQLGraph, spacy_extractor: SpacyExtractor, image_finder: ImageFinder, disambiguation: DisambiguationService,
                 amount_recommendations: int = 3):
        self._sparql_graph = sparql_graph
        self._disambiguation = disambiguation
        self._movie_extractor = MovieExtractor(spacy_extractor=spacy_extractor)
        self._image_finder = image_finder
        self._amount_recommendations = amount_recommendations

    def get_response(self, room_id: str, message: str) -> str:
        # 1) extract entities, then remove the entity from the message
        movie_names = self._movie_extractor.extract(message)
        if len(movie_names) == 0:
            return "I could not find any entity in your message, please try again and reformulate."

        movie_ids: list[str] = []
        for movie_name in movie_names:
            if self._disambiguation.multiple_uris_for_entity(category='movie', label=movie_name):
                self._disambiguation.set_ambiguity(room_id, QuestionCategory.RECOMMENDATION, 'movie', movie_name)
            else:
                movie_id = self._sparql_graph.get_uri_for_movie(movie_name).split("/")[-1]
                if movie_id == "Unknown ID":
                    return f"I don't know the movie {movie_name}. Try something else."
                movie_ids.append(movie_id)
                self._disambiguation.approve(room_id, movie_id)

        if self._disambiguation.disambiguation_required(room_id):
            return self._disambiguation.make_human_response(room_id)

        return self.get_answer_and_make_response(room_id, movie_ids)

    def get_answer_and_make_response(self, room_id: str, movie_ids: list[str]) -> str:
        self._disambiguation.clear_approved_entities(room_id)

        answer, reasoning = self._attribute_based_recommendation(movie_ids, designated_attributes=True)
        if answer:
            image = self.get_image_for_recommendation(answer)
            return self._make_human_answer(answer, reasoning=reasoning, image=image)

        answer, reasoning = self._attribute_based_recommendation(movie_ids, designated_attributes=False)
        if answer:
            image = self.get_image_for_recommendation(answer)
            return self._make_human_answer(answer, reasoning=reasoning, image=image)

        answer, reasoning = self._genre_based_recommendation(movie_ids)
        if answer:
            image = self.get_image_for_recommendation(answer)
            return self._make_human_answer(answer, reasoning=reasoning, image=image)

        movie_names = [self._sparql_graph.get_lbl_for_ent(movie_id) for movie_id in movie_ids]
        return f'There is no movie worthy of {", ".join(movie_names)} to be recommended.'

    def _attribute_based_recommendation(self, movie_ids, designated_attributes: bool) -> tuple[list | None, dict[str: str] | None]:
        if len(movie_ids) < 2:
            return None, None

        # Get common attributes of the movies
        common_attributes = self._get_common_attributes(movie_ids, designated_attributes=designated_attributes)
        if (rdflib.term.URIRef("http://www.wikidata.org/prop/direct/P179") not in common_attributes.keys() and
                len(common_attributes) < (2 if designated_attributes else 5)):
                return None, None

        # Get a movie with the same attributes
        response = self._get_movie_with_attributes(movie_ids, common_attributes)

        if not response:
            return None, None

        return response, common_attributes

    def _get_common_attributes(self, movie_ids, designated_attributes: bool) -> dict:
        movie_attributes = []
        for movie_id in movie_ids:
            """Queries the knowledge graph to get all attributes and corresponding values for a specific movie."""
            query = f"""PREFIX wd: <http://www.wikidata.org/entity/>
                        PREFIX wdt: <http://www.wikidata.org/prop/direct/>
                        SELECT ?property ?value
                        WHERE {{
                            {"VALUES ?property { wdt:P136 wdt:P57 wdt:P179 }" if designated_attributes else ""}
                            wd:{movie_id} ?property ?value .
                        }}"""

            try:
                results = self._sparql_graph.execute_query_plain_answer(query)
                attributes = {}
                for result in results:
                    property_, value = result
                    if property_ and value:
                        attributes[property_] = value
            except Exception as e:
                print(f"Error querying the graph for recommendations: {e}")
                return {}

            movie_attributes.append(attributes)

        # Start with the attributes and values of the first movie
        common_attributes = movie_attributes[0]

        # Iterate over the remaining movies
        for attributes in movie_attributes[1:]:
            # Keep only attributes that exist in the current movie and have the same value
            common_attributes = {attr: value for attr, value in common_attributes.items() if
                                 attr in attributes and attributes[attr] == value and "http" in value}

        if len(common_attributes) < 2:
            return {}

        return common_attributes

    def _get_movie_with_attributes(self, movie_ids, common_attributes) -> list:
        # Construct SPARQL filters with proper handling for IRIs and literals
        filters = "\n".join(f"?movie <{attr}> <{value}> ." for attr, value in common_attributes.items())
        query = f"""PREFIX wd: <http://www.wikidata.org/entity/>
                    PREFIX wdt: <http://www.wikidata.org/prop/direct/>
                    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
                    SELECT ?movieLabel
                    WHERE {{
                        {filters}
                        FILTER (?movie NOT IN ({", ".join(["wd:" + item for item in movie_ids])}))
                        BIND(IF(ISLITERAL(?movie), ?movie, ?movieLabel) AS ?movieLabel)
                        OPTIONAL {{
                            ?movie rdfs:label ?movieLabel .
                            FILTER(LANG(?movieLabel) = "en")
                        }}
                        OPTIONAL {{
                            ?movie wdt:P444 ?rating .
                        }}
                    }}
                    ORDER BY DESC(?rating)
                    LIMIT {self._amount_recommendations}"""

        response = self._sparql_graph.execute_query(query)
        response = [item for item in response.split("\n") if item]
        return response

    def _genre_based_recommendation(self, movie_ids) -> tuple[list[str] | None, dict[str, str] | None]:
        """Queries the knowledge graph to get genre for a specific movie."""
        query = f"""PREFIX wd: <http://www.wikidata.org/entity/>
                    PREFIX wdt: <http://www.wikidata.org/prop/direct/>
                    SELECT ?value
                    WHERE {{
                        wd:{movie_ids[0]} wdt:P136 ?value .
                    }}
                    LIMIT 1"""
        genre = self._sparql_graph.execute_query(query).split("\n")[0]

        if genre == "":
            return None, None

        query = f"""PREFIX wd: <http://www.wikidata.org/entity/>
                    PREFIX wdt: <http://www.wikidata.org/prop/direct/>
                    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
                    SELECT ?movieLabel
                    WHERE {{
                        ?movie wdt:P136 wd:{genre.split("/")[-1]} .
                        FILTER (?movie NOT IN ({", ".join(["wd:" + item for item in movie_ids])}))
                        BIND(IF(ISLITERAL(?movie), ?movie, ?movieLabel) AS ?movieLabel)
                        OPTIONAL {{
                            ?movie rdfs:label ?movieLabel .
                            FILTER(LANG(?movieLabel) = "en")
                        }}
                        OPTIONAL {{
                            ?movie wdt:P444 ?rating .
                        }}
                    }}
                    ORDER BY DESC(?rating)
                    LIMIT {self._amount_recommendations}"""

        movie = self._sparql_graph.execute_query(query)
        movie = [item for item in movie.split("\n") if item]

        return movie, {"http://www.wikidata.org/prop/direct/P136": genre}

    def _make_human_answer(self, message: list, reasoning: dict[str: str] | None = None,
                           image: str | None = None) -> str:
        if len(message) > 1:
            message_txt = ", ".join(message[:-1]) + " and " + message[-1]
        else:
            message_txt = message[0]

        arr = [f"You should definitely check out {message_txt}!",
               f"I think you'd enjoy watching {message_txt}.",
               f"Have you considered watching {message_txt}? I recommend it!",
               f"You might really like {message_txt}, give it a watch!",
               f"How about watching {message_txt}? It's worth your time.",
               f"I’d suggest you take a look at {message_txt}!",
               f"If you’re looking for something good, try {message_txt}.",
               f"You can't go wrong with {message_txt}, I highly recommend it.",
               f"I think {message_txt} would be a great pick for you to watch!",
               f"Why not give {message_txt} a try? I think you'd enjoy it!"]
        answer = random.choice(arr)

        # if settings.use_llm:
        #     try:
        #         answer = LlmService().beautify_answer(answer)
        #     except Exception as e:
        #         print(f"Error in LLM: {e}")

        if reasoning:
            list_of_attributes = '\n - '.join(f'{self._sparql_graph.get_lbl_for_ent(str(key))}: {self._sparql_graph.get_lbl_for_ent(str(value))}' for key, value in reasoning.items())
            answer = (f"{answer}\n\nYou wonder why? They share the following attributes:\n"
                      f" - {list_of_attributes}")

        if image:
            answer = f"{image}\n{answer}"

        return answer

    def get_image_for_recommendation(self, movie_labels: list) -> str | None:
        for movie_label in movie_labels:
            movie_uri = self._sparql_graph.get_ent_for_lbl(movie_label).split("/")[-1]
            if movie_uri == "Unknown Entity":
                continue

            image = self._image_finder.get_image_for_movie(movie_uri)
            if image:
                return image

        return None
