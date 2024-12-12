import logging
from abc import ABC, abstractmethod
from app.services.sparql_graph import SPARQLGraph

logger = logging.getLogger(__name__)


class AnsweringService(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def get_response(self, room_id: str, message: str) -> str:
        pass

class SPARQLAnswerService(AnsweringService):
    def __init__(self, sparql_graph: SPARQLGraph):
        self.sparql_graph: SPARQLGraph = sparql_graph

    def get_response(self, query: str, room_id: str = "") -> str:
        """Use the SPARQLGraph to execute a query and return the response."""
        return self.sparql_graph.execute_query(query)
