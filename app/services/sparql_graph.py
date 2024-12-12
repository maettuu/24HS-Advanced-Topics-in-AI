import json
from rdflib import Graph
from datetime import datetime
import os

from app.config.enums import Environment
from app.config.app import settings


class SPARQLGraph:
    def __init__(self, env: Environment, lazy_load=False):
        self.graph = None
        self.ent2lbl = {}
        self.lbl2ent = {}
        self.rel2lbl = {}
        self.lbl2rel = {}

        if env == Environment.DEV:
            self.rdf_file = os.sep.join((settings.utils_path, "useful_dataset", "graph_test.nt"))
            self.metadata_path = os.sep.join((settings.utils_path, "useful_dataset", "graph"))
        else:
            self.rdf_file = os.sep.join((settings.utils_path, "too_large_dataset", "ddis-movie-graph.nt"))
            self.metadata_path = os.sep.join((settings.utils_path, "useful_dataset", "graph"))

        # if RDF file does not exist, raise an error
        if not os.path.exists(self.rdf_file):
            raise FileNotFoundError(f"RDF file not found: {self.rdf_file}")

        # Load JSON metadata files
        self._load_metadata()

        # Load graph if not lazy loading
        if not lazy_load:
            self._load_graph()

    def _load_graph(self):
        """Load RDF graph on initialization."""
        graph = Graph()
        start = datetime.now()
        print("Initializing SPARQLGraph")
        try:
            graph.parse(self.rdf_file, format='turtle')
            print(f"Graph loaded with {len(graph)} triples after {datetime.now() - start}")
        except Exception as e:
            print(f"Failed to load RDF data: {e}")
        self.graph = graph

    def _load_metadata(self):
        """Load metadata from JSON files."""
        try:
            with open(os.path.join(self.metadata_path, 'ent2lbl.json'), 'r') as file:
                self.ent2lbl = json.load(file)

            with open(os.path.join(self.metadata_path, 'lbl2ent.json'), 'r') as file:
                self.lbl2ent = json.load(file)

            with open(os.path.join(self.metadata_path, 'rel2lbl.json'), 'r') as file:
                self.rel2lbl = json.load(file)

            with open(os.path.join(self.metadata_path, 'lbl2rel.json'), 'r') as file:
                self.lbl2rel = json.load(file)

            with open(os.path.join(self.metadata_path, 'movie2id.json'), 'r') as file:
                self.movie2id = json.load(file)

            with open(os.path.join(self.metadata_path, 'person2id.json'), 'r') as file:
                self.person2id = json.load(file)

            print("Metadata loaded successfully from JSON files.")
        except FileNotFoundError as e:
            print(f"Metadata JSON file not found: {e}")
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON metadata file: {e}")

    def execute_query_plain_answer(self, query: str):
        """Run a SPARQL query on the loaded graph."""
        try:
            # Lazy-load the graph if it wasn't loaded initially
            if not self.graph:
                self._load_graph()

            return self.graph.query(query)
        except Exception as e:
            return str(e)

    def execute_query(self, query: str) -> str:
        """Run a SPARQL query on the loaded graph."""
        try:
            # Lazy-load the graph if it wasn't loaded initially
            if not self.graph:
                self._load_graph()

            result = self.graph.query(query)
            results_list = []
            for row in result:
                results_list.append([str(item).encode("utf-8").decode("utf-8") for item in row])

            # Join results into a string with newlines
            return '\n'.join(['\t'.join(row) for row in results_list])

        except Exception as e:
            print(f"Error querying the graph: {e}")
            return ""

    # Metadata access methods
    def get_lbl_for_ent(self, entity_uri: str) -> str:
        """Get label for a given entity URI."""
        return self.ent2lbl.get(entity_uri, "Unknown Label")

    def get_ent_for_lbl(self, label: str) -> str:
        """Get entity URI for a given label."""
        return self.lbl2ent.get(label, "Unknown Entity")

    def get_lbl_for_rel(self, relation_uri: str) -> str:
        """Get label for a given relation URI."""
        return self.rel2lbl.get(relation_uri, "Unknown Label")

    def get_uri_for_movie(self, movie: str) -> str:
        """Get movie URI for a given movie label"""
        return self.movie2id.get(movie, "Unknown ID")

    def get_rel_for_lbl(self, label: str) -> str:
        """Get relation URI for a given label."""
        return self.lbl2rel.get(label, "Unknown Relation")

    def get_entities_labels(self):
        """Get all entities from the graph."""
        return list(self.lbl2ent.keys())

    def get_relations_labels(self):
        """Get all relations from the graph."""
        return list(self.lbl2rel.keys())

    def get_uri_for_person(self, person: str) -> str:
        """Get the IMDB URI for a given person label."""
        return self.person2id.get(person, "Unknown ID")
