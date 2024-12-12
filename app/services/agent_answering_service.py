import logging
import numpy as np

from app.config.app import settings
from app.config.enums import Environment, Milestone
from app.services.crowd_service import CrowdService
from app.services.disambiguation_service import DisambiguationService
from app.services.embeddings_service import EmbeddingsService
from app.services.extractors.main import SpacyExtractor
from app.services.knowledge_answering_service import EmbeddingAndKnowledgeAnswerService
from app.services.llm_service import LlmService
from app.services.image_finder import ImageFinder
from app.services.question_classifier import QuestionCategory, QuestionClassifier
from app.services.recommendation_service import RecommendationService
from app.services.sparql_graph import SPARQLGraph

logger = logging.getLogger(__name__)


class AgentAnsweringService:
    def __init__(self, environment: Environment = settings.environment):
        print("Initializing AgentAnsweringService with environment: ", environment)
        lazy_load = environment != Environment.PROD
        sparql_graph = SPARQLGraph(environment, lazy_load)
        embeddings = EmbeddingsService(lazy_load)
        crowd = CrowdService()

        self.disambiguation = DisambiguationService()
        self.message_received_templates = [
            "Good question, let's see.",
            "I hear you, let me quickly have a look.",
            "Interesting query, I'm on it!",
            "Hmm, checking now.",
            "Great question! Let me find out for you.",
            "Got it! Let me dive into that.",
            "Okay, give me a moment to explore this for you.",
            "I’m on it! Just a second.",
            "Let me see what I can find for you.",
            "Interesting thought! Let me check.",
            "One moment while I work on that.",
            "I’ll take a closer look at this for you.",
            "Checking now, hold tight!",
            "Let me dig into that for you.",
            "On it! This won’t take long.",
            "Let me explore this for you.",
            "Thanks for your patience! I’m checking now.",
            "Hang on while I gather some details.",
            "Let’s figure this out together. One sec!",
            "Great topic! Let me do a quick search.",
            "This is intriguing! Let me find the answer.",
            "Give me a moment to uncover the details.",
            "I’ll have some info for you shortly.",
            "Let me fetch the details for you."
        ]

        if settings.milestone == Milestone.ONE:
            raise NotImplementedError("Milestone 1 not supported anymore")
            # answering_service = SPARQLAnswerService(sparql_graph)
        elif settings.milestone == Milestone.TWO:
            raise NotImplementedError("Milestone 2 not supported anymore")
            # answering_service = EmbeddingAndKnowledgeAnswerService(sparql_graph, embeddings)
        elif settings.milestone == Milestone.THREE:
            spacy_extractor = SpacyExtractor()
            print("Spacy extractor initialized")
            self.knowledge_answering_service = EmbeddingAndKnowledgeAnswerService(sparql_graph, embeddings, crowd, self.disambiguation, spacy_extractor)
            self.image_finder = ImageFinder(sparql_graph, self.disambiguation, spacy_extractor)
            self.recommendation_service = RecommendationService(sparql_graph, spacy_extractor, self.image_finder, self.disambiguation)
        else:  # Final Project
            raise NotImplementedError("Final Project not implemented yet")

        self.question_classifier = QuestionClassifier()

    def disambiguation_required(self, room_id: str) -> bool:
        return self.disambiguation.disambiguation_required(room_id)

    def get_message_received_template(self) -> str:
        return np.random.choice(self.message_received_templates)

    def get_message_picked_choice(self, message: str, room_id: str = "test_id") -> str | None:
        entity = self.disambiguation.extract_choice_and_retrieve_entity(room_id, message)
        if not entity or entity == "exit":
            return None
        entity_uri = "http://www.wikidata.org/entity/" + entity
        ambiguity = self.disambiguation.get_ambiguity(room_id)
        description = self.disambiguation.get_description_for_entity_uri(ambiguity["category"], ambiguity["entity_label"], entity_uri)
        return f"I understand you\'re asking about the {description}."

    def message_is_smalltalk(self, message: str) -> bool:
        return self.question_classifier.classify(message) == QuestionCategory.SMALLTALK

    def get_answer_for_message(self, message: str, room_id: str = "test_id") -> str:
        """
        Determines the appropriate response for a given message.
        """
        if self.disambiguation_required(room_id):
            return self.get_clarification_for_ambiguity(room_id, message)

        question_type: QuestionCategory = self.question_classifier.classify(message)
        print(f"Question type: {question_type}")

        if question_type == QuestionCategory.SMALLTALK:
            print(f"Use LLM: {settings.use_llm}")
            if settings.use_llm:
                try:
                    return LlmService().small_talk(message)
                except Exception as e:
                    print(f"Error in LLM: {e}")

            if self._is_welcome_message(message):
                return "Hi! Nice to have you here. I can answer multiple questions regarding movies. Let's start!"
            if self._is_end_message(message):
                return "Goodbye! I hope I was able to help you. Feel free to come back anytime."

        elif question_type == QuestionCategory.KNOWLEDGE:
            return self.knowledge_answering_service.get_response(room_id, message)
        elif question_type == QuestionCategory.MULTIMEDIA:
            return self.image_finder.get_response(room_id, message)
        elif question_type == QuestionCategory.RECOMMENDATION:
            return self.recommendation_service.get_response(room_id, message)

        return "I'm not sure how to answer that. Can you please rephrase?"

    def get_clarification_for_ambiguity(self, room_id: str, message: str) -> str:
        """
        Determines the appropriate response for a given ambiguity.
        """
        entity = self.disambiguation.extract_choice_and_retrieve_entity(room_id, message)
        if not entity:
            return "I'm not sure which one you meant." \
                   " Please specify using the list number or the full description.\n" \
                   "Let me know if I should stop the disambiguation."
        if entity == "exit":
            self.disambiguation.clear_ambiguities(room_id)
            return "Okay, on your demand the process has been interrupted. I\'m happy to answer any other questions :)"
        ambiguity = self.disambiguation.get_ambiguity(room_id)
        self.disambiguation.remove_ambiguity(room_id)
        if ambiguity["question_type"] == QuestionCategory.KNOWLEDGE:
            return self.knowledge_answering_service.get_answer_and_make_response(
                entity,
                ambiguity["relation"],
                ambiguity["entity_label"],
                ambiguity["relation_label"]
            )
        elif ambiguity["question_type"] == QuestionCategory.MULTIMEDIA:
            self.disambiguation.remove_ambiguity(room_id)
            return self.image_finder.get_answer_and_make_response(
                entity,
                self.image_finder.get_image_for_imdb_id,
                ambiguity["entity_label"],
                ambiguity["category"]
            )
        elif ambiguity["question_type"] == QuestionCategory.RECOMMENDATION:
            self.disambiguation.approve(room_id, entity)
            if self.disambiguation_required(room_id):
                return "Unfortunately, I need more information still. " + self.disambiguation.make_human_response(room_id)
            return self.recommendation_service.get_answer_and_make_response(room_id, self.disambiguation.get_approved_entities(room_id))
        return "I'm not sure how to answer that. Can you please rephrase?"

    @staticmethod
    def _is_welcome_message(message: str) -> bool:
        # Expanded list of possible greetings
        greetings = [
            "hi", "hello", "hey", "greetings", "heyo", "hallo", "hi there", "hello there", "hey there",
            "morning", "good morning", "hiya", "what's up", "sup", "yo", "howdy", "good day",
            "good afternoon", "good evening", "evening", "g'day", "salutations"
        ]

        # Clean the message for consistent checking
        message_cleaned = message.lower().strip()

        # Check if any greeting matches and is short enough relative to the full message
        for greeting in greetings:
            greeting_cleaned = greeting.lower()

            if message_cleaned.startswith(greeting_cleaned) and len(greeting_cleaned) * 2 >= len(message_cleaned):
                return True

        return False

    @staticmethod
    def _is_end_message(messages: str) -> bool:
        matches = ["bye", "goodbye", "farewell", "adios", "ciao", "later", "see you", "talk to you later",
                   "ttyl", "goodnight", "night", "sleep", "rest alright thanks", "alright thank you", "thanks",
                   "thank you", "ok thanks", "ok thank you", "okay thanks", "okay thank you", "ok bye", "okay bye",
                   "ok goodbye", "okay goodbye"]
        cleaned_messages = messages.lower().strip()
        for match in matches:
            ending_cleaned = match.lower()
            if cleaned_messages.startswith(ending_cleaned) and len(ending_cleaned) * 2 >= len(cleaned_messages):
                return True

        return False
