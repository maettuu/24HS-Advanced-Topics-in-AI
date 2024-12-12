import csv
import os
from datetime import datetime

import numpy as np
import rdflib
from rdflib import URIRef
from sklearn.metrics import pairwise_distances

from app.config.app import settings


class EmbeddingsService:
    def __init__(self, lazy_load=False):
        self.entity_emb: np.ndarray | None = None
        self.relation_emb: np.ndarray | None = None
        self.ent2id = {}
        self.rel2id = {}
        self.id2ent = {}
        self.id2rel = {}

        # Load metadata from DEL files
        self._load_metadata()

        # Load embeddings if not lazy loading
        if not lazy_load:
            self._load_embeddings()

    def _load_embeddings(self):
        """Load embeddings on initialization."""
        start = datetime.now()
        print("Initializing Embeddings")
        try:
            self.entity_emb = np.load(
                os.sep.join((settings.utils_path, "too_large_dataset", "ddis-graph-embeddings", "entity_embeds.npy")))
            self.relation_emb = np.load(
                os.sep.join((settings.utils_path, "too_large_dataset", "ddis-graph-embeddings", "relation_embeds.npy")))

            print(f"Embeddings loaded with {len(self.entity_emb)} entries for entities and {len(self.relation_emb)} "
                  f"entries for relations after {datetime.now() - start}")
        except Exception as e:
            print(f"Failed to load NPY data: {e}")

    def _load_metadata(self):
        """Load metadata from DEL files."""
        try:
            with open(
                    os.sep.join((settings.utils_path, "too_large_dataset", "ddis-graph-embeddings", "entity_ids.del")),
                    'r') as ifile:
                self.ent2id = {rdflib.term.URIRef(ent): int(idx) for idx, ent in csv.reader(ifile, delimiter='\t')}
                self.id2ent = {v: k for k, v in self.ent2id.items()}
            with open(os.sep.join(
                    (settings.utils_path, "too_large_dataset", "ddis-graph-embeddings", "relation_ids.del"))) as ifile:
                self.rel2id = {rdflib.term.URIRef(rel): int(idx) for idx, rel in csv.reader(ifile, delimiter='\t')}
                self.id2rel = {v: k for k, v in self.rel2id.items()}

            print("Metadata loaded successfully from DEL files.")
        except FileNotFoundError as e:
            print(f"Metadata JSON file not found: {e}")

    def calculate_embeddings(self, entity: str, relation: str) -> str:
        """Calculate the result from embeddings."""
        WD = rdflib.Namespace('http://www.wikidata.org/entity/')
        WDT = rdflib.Namespace('http://www.wikidata.org/prop/direct/')
        DDIS = rdflib.Namespace('http://ddis.ch/atai/')
        RDFS = rdflib.namespace.RDFS
        SCHEMA = rdflib.Namespace('http://schema.org/')

        try:
            # Lazy-load the graph if it wasn't loaded initially
            if self.entity_emb is None or self.relation_emb is None:
                self._load_embeddings()

            # Calculate embeddings
            head = self.entity_emb[self.ent2id[URIRef(WD + entity)]]
            pred = self.relation_emb[self.rel2id[URIRef(WDT + relation)]]
            lhs = head + pred  # add vectors according to TransE scoring function.
            dist = pairwise_distances(lhs.reshape(1, -1), self.entity_emb).reshape(-1)  # compute distance to any entity
            most_likely = dist.argsort()  # find most plausible entities

            result = self.get_ent_for_id(most_likely[0])

            return result
        except Exception as e:
            print(str(e))
            return "While calculating embeddings, an error occurred."

    # Metadata access methods
    def get_id_for_ent(self, entity_uri: str) -> int:
        """Get id for a given entity URI."""
        entity_uri = URIRef(entity_uri)
        return self.ent2id.get(entity_uri, -1)

    def get_ent_for_id(self, _id: int) -> str:
        """Get entity URI for a given id."""
        return str(self.id2ent.get(_id, "Unknown Entity"))

    def get_id_for_rel(self, relation_uri: str) -> int:
        """Get id for a given relation URI."""
        relation_uri = URIRef(relation_uri)
        return self.rel2id.get(relation_uri, -1)

    def get_rel_for_id(self, _id: int) -> str:
        """Get relation URI for a given id."""
        return str(self.id2rel.get(_id, "Unknown Relation"))

    def get_entities_ids(self) -> list:
        """Get all entity ids from the embeddings."""
        return list(self.id2ent.keys())

    def get_relations_ids(self) -> list:
        """Get all relation ids from the embeddings."""
        return list(self.id2rel.keys())
