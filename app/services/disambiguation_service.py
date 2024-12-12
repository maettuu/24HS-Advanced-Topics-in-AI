import os
import json

from collections import defaultdict

from app.config.app import settings
from app.services.question_classifier import QuestionCategory


class DisambiguationService:
    def __init__(self):
        self.entity_metadata = {}
        self.metadata_path = os.sep.join((settings.utils_path, "useful_dataset", "graph"))
        self.current_ambiguities = defaultdict(list)
        self.approved_entities = defaultdict(list)

        # Load metadata from JSON file
        self._load_metadata()

    def _load_metadata(self):
        """Load metadata from JSON files."""
        print("Initializing Disambiguation")
        try:
            with open(os.path.join(self.metadata_path, 'movie2ids.json'), 'r') as file:
                self.entity_metadata['movie'] = json.load(file)

            with open(os.path.join(self.metadata_path, 'person2ids.json'), 'r') as file:
                self.entity_metadata['person'] = json.load(file)

            print("Metadata loaded successfully from JSON files.")
        except FileNotFoundError as e:
            print(f"Metadata JSON file not found: {e}")
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON metadata file: {e}")

    def disambiguation_required(self, room_id: str) -> bool:
        """Checks if there currently is an ambiguity which requires clarification."""
        return len(self.current_ambiguities[room_id]) > 0

    def set_ambiguity(self,
                      room_id: str,
                      question_type: QuestionCategory,
                      category: str,
                      entity_label: str,
                      relation: str = "",
                      relation_label: str = "") -> None:
        """Set current ambiguity which requires clarification."""
        self.current_ambiguities[room_id].append({
            "question_type": question_type,
            "category": category,
            "entity_label": entity_label,
            "relation": relation,
            "relation_label": relation_label
        })

    def get_ambiguity(self, room_id: str) -> dict[str, QuestionCategory | str]:
        """Retrieve current ambiguity details."""
        return self.current_ambiguities[room_id][0].copy()

    def remove_ambiguity(self, room_id: str) -> None:
        """Removes current ambiguity."""
        self.current_ambiguities[room_id] = self.current_ambiguities[room_id][1:]

    def clear_ambiguities(self, room_id: str) -> None:
        self.current_ambiguities[room_id].clear()

    def approve(self, room_id: str, entity_uri: str) -> None:
        """Adds entity to list of approved entities with resolved or no ambiguity."""
        self.approved_entities[room_id].append(entity_uri)

    def get_approved_entities(self, room_id: str) -> list[str]:
        """Gets all approved entities."""
        return self.approved_entities[room_id].copy()

    def clear_approved_entities(self, room_id: str) -> None:
        """Clears list of approved entities."""
        self.approved_entities[room_id].clear()

    def extract_choice_and_retrieve_entity(self, room_id: str, message: str) -> str | None:
        """Resolve ambiguity based on user input and return the URI of the selected entity."""
        choice = self.extract_choice(room_id, message)
        if choice is None:
            return None
        return self.get_id_for_entity(
            self.current_ambiguities[room_id][0]["category"],
            self.current_ambiguities[room_id][0]["entity_label"],
            choice)

    def extract_choice(self, room_id: str, message: str) -> int | None:
        """Parses message and extracts user choice."""
        possible_choices = {
            0: ['1', '1st', 'one', 'first'],
            1: ['2', '2nd', 'two', 'second'],
            2: ['3', '3rd', 'three', 'third'],
            3: ['4', '4th', 'four', 'fourth'],
            4: ['5', '5th', 'five', 'fifth'],
            5: ['6', '6th', 'six', 'sixth'],
            'exit': ['exit', 'done', 'stop', 'not', 'no', 'interrupt', 'good', 'anymore', 'already', 'enough']
        }

        cleaned_message = message.lower().strip().replace(".", "").split()
        if "last" in cleaned_message:
            return -1
        for word in cleaned_message:
            for idx, phrases in possible_choices.items():
                if word in phrases:
                    return idx

        descriptions = self.get_descriptions_for_entity(
            self.current_ambiguities[room_id][0]["category"],
            self.current_ambiguities[room_id][0]["entity_label"]
        )
        for idx, description in enumerate(descriptions):
            if message in description:
                return idx
            if message in f"{idx+1}. {description}":
                return idx
        return None

    def multiple_uris_for_entity(self, category: str, label: str) -> bool:
        """Check if there are multiple uris for a given entity."""
        return label in self.entity_metadata[category]

    def make_human_response(self, room_id: str) -> str:
        label = self.current_ambiguities[room_id][0]["entity_label"]
        descriptions = self.get_descriptions_for_entity(self.current_ambiguities[room_id][0]["category"], label)
        human_response = f"It seems there are several entries for {label}:\n"
        for idx, desc in enumerate(descriptions, start=1):
            human_response += f"{idx}. {desc}\n"
        human_response += "Let me know which one you\'re asking about or whether you\'d like to stop."
        return human_response

    # Metadata access methods
    def get_uris_for_entity(self, category: str, label: str) -> list[str] | None:
        """Get all entity URIs for a given label."""
        return list(self.entity_metadata[category].get(label, {}).keys()) or None

    def get_id_for_entity(self, category: str, label: str, idx: int | str) -> str | None:
        """Get entity URI for a given label and index."""
        if idx == "exit":
            return "exit"
        uris = self.get_uris_for_entity(category, label)
        if uris and -1 <= idx < len(uris):
            return uris[idx].split("/")[-1]
        return None

    def get_descriptions_for_entity(self, category: str, label: str) -> list[str] | None:
        """Get all entity descriptions for a given label."""
        return list(self.entity_metadata[category].get(label, {}).values()) or None

    def get_description_for_entity_uri(self, category: str, label: str, uri: str) -> str | None:
        """Get entity description for a given label and URI."""
        return self.entity_metadata[category].get(label, {}).get(uri)
