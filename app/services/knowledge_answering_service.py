import logging
import random

from app.config.app import settings
from app.services.crowd_service import CrowdService
from app.services.disambiguation_service import DisambiguationService
from app.services.embeddings_service import EmbeddingsService
from app.services.extractors.movie_extraction import MovieExtractor
from app.services.extractors.relationship_extraction import RelationshipExtractor
from app.services.llm_service import LlmService
from app.services.question_classifier import QuestionCategory
from app.services.sparql_graph import SPARQLGraph
from app.services.extractors.main import SpacyExtractor
from app.services.answering_service import AnsweringService, SPARQLAnswerService

logger = logging.getLogger(__name__)


class EmbeddingAndKnowledgeAnswerService(AnsweringService):
    def __init__(
            self,
            sparql_graph: SPARQLGraph,
            embeddings: EmbeddingsService,
            crowd: CrowdService,
            disambiguation: DisambiguationService,
            spacy_extractor: 'SpacyExtractor' = None):
        self._sparql_graph: SPARQLGraph = sparql_graph
        self._embeddings: EmbeddingsService = embeddings
        self._crowd: CrowdService = crowd
        self._disambiguation: DisambiguationService = disambiguation
        self._llm = LlmService()
        self._movie_extractor = MovieExtractor(spacy_extractor=spacy_extractor)
        self._relationship_extractor = RelationshipExtractor(spacy_extractor=spacy_extractor)
        self._answering_service = SPARQLAnswerService(self._sparql_graph)

    def get_response(self, room_id: str, message: str) -> str:
        """Extract entities from the message and query the SPARQL graph."""
        # TODO implement a safety mechanisms when one of them is none or it fails, then run the embeddings based

        clean_message = clean_query(message)

        # 1) extract entities, then remove the entity from the message
        entities = self._movie_extractor.extract(clean_message)
        if len(entities) == 0:
            return "I could not find any entity in your message, please try again and reformulate."
        if len(entities) > 1:
            return "I found multiple entities in your message, please try again and ask about one movie only!"

        entity_label = entities[0]
        clean_message = clean_message.replace(entity_label, "")

        # 2) extract relation
        relations = self._relationship_extractor.extract(clean_message)
        if len(relations) == 0:
            return "I could not find any relation in your message, please try again and reformulate."
        if len(relations) > 1:
            return "I found multiple relations in your message, please try again and ask about one relation only!"

        relation_label = relations[0]

        # log found entity and relation label
        logging.info(f"Movie: {entity_label}, Relation: {relation_label}")

        relation = self._sparql_graph.get_rel_for_lbl(relation_label).split("/")[-1]

        # 3) check for label ambiguity
        if self._disambiguation.multiple_uris_for_entity(category='movie', label=entity_label):
            self._disambiguation.set_ambiguity(room_id, QuestionCategory.KNOWLEDGE, 'movie', entity_label, relation=relation, relation_label=relation_label)
            return self._disambiguation.make_human_response(room_id)

        entity = self._sparql_graph.get_uri_for_movie(entity_label).split("/")[-1]
        if entity == "Unknown ID":
            return f"I don't know the movie {entity_label}. Try something else."
        return self.get_answer_and_make_response(entity, relation, entity_label, relation_label)

    def get_answer_and_make_response(self, entity: str, relation: str, entity_label: str, relation_label: str) -> str:
        # print(f"Movie: {entity}, Relation: {relation}")
        logging.info(f"Movie: {entity}, Relation: {relation}")

        # 4) Check crowd data and return if we have an answer
        answer_crowd = self.get_answer_from_crowd(entity, relation)
        if answer_crowd:
            return self.make_human_response(entity_label, relation_label, answer_crowd, source="crowd")

        # 5) Query the graph and return if we have an answer
        answer_graph = self.get_answer_from_graph(entity, relation)
        if answer_graph:
            return self.make_human_response(entity_label, relation_label, answer_graph)

        # 6) If we don't have an answer from the graph, try embeddings
        answer_embeddings = self.get_answer_from_embeddings(entity, relation)
        if answer_embeddings:
            return self.make_human_response(entity_label, relation_label, answer_embeddings, source="embeddings")

        return ("I could not find an answer to your question or calculate it from the embeddings. Please try "
                "rephrasing and ask again.")

    def get_answer_from_graph(self, entity: str, relation: str = None) -> str:
        query = f"""PREFIX wd: <http://www.wikidata.org/entity/>
                    PREFIX wdt: <http://www.wikidata.org/prop/direct/>
                    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
                    SELECT ?answerLabel
                    WHERE {{
                        wd:{entity} wdt:{relation} ?answer .
                        BIND(IF(ISLITERAL(?answer), ?answer, ?answerLabel) AS ?answerLabel)
                        OPTIONAL {{
                            ?answer rdfs:label ?answerLabel .
                            FILTER(LANG(?answerLabel) = "en")
                        }}
                    }}
                    LIMIT 1"""

        response = self._sparql_graph.execute_query(query)
        return response

    def get_answer_from_embeddings(self, entity: str, relation: str) -> str:
        response = self._embeddings.calculate_embeddings(entity, relation)
        response = self._sparql_graph.get_lbl_for_ent(response)

        if response == "Unknown Label":
            return ""

        return response

    def get_answer_from_crowd(self, entity: str, relation: str) -> dict[str, str | int | float] | None:
        response = self._crowd.get_crowd_data_for_ent_rel(entity, relation)
        if response:
            label = self._sparql_graph.get_lbl_for_ent("http://www.wikidata.org/entity/" + response.get("value"))
            if label == "Unknown Label":
                return response
            response["value"] = label
            return response
        return None

    def make_human_response(self, entity: str, relation: str, prediction: str | dict[str, str | int | float],
                            source: str = "graph") -> str:
        answer_source = ""
        if source == "crowd":
            crowd_data = prediction.copy()
            prediction = crowd_data.get("value")
            answer_source = ' (This response was populated using crowd-sourced data,' \
                            f' which shows an inter-rater agreement of {crowd_data.get("fleiss_kappa")} and an' \
                            f' answer distribution of {crowd_data.get("n_correct")} supporting' \
                            f' and {crowd_data.get("n_incorrect")} rejecting votes.)'
        elif source == "embeddings":
            answer_source = " (Embedding Answer)"

        # if prediction has format yyyy-mm-dd, convert it to a year only
        if len(prediction) == 10 and prediction[4] == "-" and prediction[7] == "-":
            prediction = prediction[:4]

        arr = [f"I think it is {prediction}.",
               f"That is a good question, I think that the answer is {prediction}.",
               f"As far as I know, it is {prediction}.",
               f"I would say that it is {prediction}.",
               f"According to my knowledge, it is {prediction}.",
               f"I'm almost certain that it is {prediction}.",

               # more fancy sentences -> they might not work all the time
               f"The {relation} is {prediction} for {entity}.",
               f"I think that the {relation} is {prediction} for {entity}.",
               f"I would say that the {relation} of {entity} is {prediction}.", ]

        if settings.use_llm:
            try:
                answer = self._llm.beautify_answer(f"The {relation} of {entity} is {prediction}.")
            except:  # Just in case the LLM service fails
                answer = random.choice(arr)
        else:
            answer = random.choice(arr)

        # do a random choice
        return f'{answer}{answer_source}'


def clean_query(query: str) -> str:
    # Replace multiple spaces with a single space
    query = " ".join(query.split())
    # Trim leading and trailing spaces
    query = query.strip()

    # List of characters to remove if they are the last character in the query
    remove_chars = ["?", ".", ",", "!", "\"", "'"]

    # Remove the last character if it is in the list of remove_chars
    if query and query[-1] in remove_chars:
        query = query[:-1]

    return query
