import os

import pandas as pd

from app.config.app import settings


class CrowdService:
    def __init__(self):
        self.crowd_data = {}

        # Load metadata from CSV file
        self._load_metadata()

    @staticmethod
    def _truncate_to_three_decimals(value):
        return int(value * 1000) / 1000

    def _load_metadata(self):
        """Load metadata from CSV file."""
        print("Initializing Crowd Data")
        try:
            metadata = pd.read_csv(os.sep.join((settings.utils_path, "useful_dataset", "crowd", "processed_crowd.csv")))
            for col in ["Input1ID", "Input2ID", "Input3ID"]:
                metadata[col] = metadata[col].str.replace("wd:", "").str.replace("wdt:", "").str.replace("ddis:", "")
            for _, row in metadata.iterrows():
                entity = row["Input1ID"]
                relation = row["Input2ID"]
                if entity not in self.crowd_data:
                    self.crowd_data[entity] = {}
                self.crowd_data[entity][relation] = {
                    "value": row["Input3ID"],
                    "n_correct": int(row["nr_correct"]),
                    "n_incorrect": int(row["nr_incorrect"]),
                    "fleiss_kappa": self._truncate_to_three_decimals(float(row["Fleiss Kappa"]))
                }

            print("Metadata loaded successfully from CSV file.")
        except FileNotFoundError as e:
            print(f"Metadata CSV file not found: {e}")

    # Metadata access methods
    def get_crowd_data_for_ent(self, entity_uri: str) -> dict[str, dict[str, str | int | float]] | None:
        """Get all crowdsourced data for a given entity URI."""
        return self.crowd_data.get(entity_uri)

    def get_crowd_data_for_ent_rel(self, entity_uri: str, relation_uri: str) -> dict[str, str | int | float] | None:
        """Get crowdsourced data for given entity and relation URI."""
        return self.crowd_data.get(entity_uri, {}).get(relation_uri)

    def get_crowd_data_value_for_ent_rel(self, entity_uri: str, relation_uri: str) -> str:
        """Get crowdsourced data value for given entity and relation URI."""
        return self.crowd_data.get(entity_uri, {}).get(relation_uri, {}).get("value") or "Unknown value"

    def get_crowd_data_support_votes_for_ent_rel(self, entity_uri: str, relation_uri: str) -> int:
        """Get crowdsourced data value for given entity and relation URI."""
        return self.crowd_data.get(entity_uri, {}).get(relation_uri, {}).get("n_correct") or -1

    def get_crowd_data_reject_votes_for_ent_rel(self, entity_uri: str, relation_uri: str) -> int:
        """Get crowdsourced data value for given entity and relation URI."""
        return self.crowd_data.get(entity_uri, {}).get(relation_uri, {}).get("n_incorrect") or -1

    def get_crowd_data_fleiss_kappa_for_ent_rel(self, entity_uri: str, relation_uri: str) -> float:
        """Get crowdsourced data value for given entity and relation URI."""
        return self.crowd_data.get(entity_uri, {}).get(relation_uri, {}).get("fleiss_kappa") or 0.0

    def get_crowd_entity_ids(self) -> list:
        """Get all entity ids from the crowd data."""
        return list(self.crowd_data.keys())
