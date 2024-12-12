import asyncio
import logging
import time
from typing import List

from fastapi import FastAPI
from speakeasypy import Chatroom, Speakeasy  # installed as wheel

from app.config.app import settings
from app.config.enums import Environment
from app.services.agent_answering_service import AgentAnsweringService


# Logging setup
def setup_logging(log_file=f'app.{str(settings.environment.value)}.log'):
    logging.basicConfig(level=logging.DEBUG if settings.environment == Environment.DEV else logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s",
                        handlers=[logging.StreamHandler(),  # Console output
                                  logging.FileHandler(log_file)  # File output
                                  ])



# Initialize logging
setup_logging()
logger = logging.getLogger(__name__)

app = FastAPI()


class Agent:
    def __init__(self, username: str, password: str, agent_answering_service: "AgentAnsweringService"):

        self.username = username
        self.speakeasy = Speakeasy(host=settings.default_host_url, username=username, password=password)
        self.speakeasy.login()

        self.agent_answering_service = agent_answering_service

    async def listen(self):
        while True:
            rooms: List[Chatroom] = self.speakeasy.get_rooms(active=True)
            for room in rooms:
                if not room.initiated:
                    room.post_messages(f'Hello and welcome! This is {room.my_alias}. I\'m happy to answer your questions :)')
                    room.initiated = True

                for message in room.get_messages(only_partner=True, only_new=True):
                    print(f"\t- Chatroom {room.room_id} "
                          f"- new message #{message.ordinal}: '{message.message}' "
                          f"- {self.get_time()}")

                    logger.info(f"Received message: {message.message}")

                    if self.agent_answering_service.disambiguation_required(room.room_id):
                        message_picked_choice = self.agent_answering_service.get_message_picked_choice(message.message, room.room_id)
                        if message_picked_choice:
                            message_picked_choice = str(message_picked_choice).encode("utf-8").decode("latin-1")
                            room.post_messages(message_picked_choice)
                    elif not self.agent_answering_service.message_is_smalltalk(message.message):
                        message_received_str = self.agent_answering_service.get_message_received_template()
                        message_received_str = str(message_received_str).encode("utf-8").decode("latin-1")
                        room.post_messages(message_received_str)

                    message_str = self.agent_answering_service.get_answer_for_message(message.message, room.room_id)

                    logging.info(f"Response: {str(message_str)}")

                    message_str = str(message_str).encode("utf-8").decode("latin-1")
                    room.post_messages(message_str)
                    room.mark_as_processed(message)

                for reaction in room.get_reactions(only_new=True):
                    print(f"\t- Chatroom {room.room_id} "
                          f"- new reaction #{reaction.message_ordinal}: '{reaction.type}' "
                          f"- {self.get_time()}")
                    # Your agent logic here
                    room.post_messages(f"Received your reaction: '{reaction.type}' ")
                    room.mark_as_processed(reaction)

            await asyncio.sleep(settings.listen_freq)

    @staticmethod
    def get_time() -> str:
        return time.strftime("%H:%M:%S, %d-%m-%Y", time.localtime())


@app.on_event("startup")
async def startup_event():
    global answering_service

    print(f"Starting up in mode: {settings.environment}")

    answering_service = AgentAnsweringService()

    agent_instance = Agent(settings.bot_username, settings.bot_password, answering_service)

    asyncio.create_task(agent_instance.listen())


@app.get("/")
def read_root():
    return {"message": "Welcome to the Chatbot!"}


@app.get("/{message}")
def read_root(message: str):
    return answering_service.get_answer_for_message(message)
