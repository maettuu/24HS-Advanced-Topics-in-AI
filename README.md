# 24HS Advanced Topics in Artificial Intelligence (ATAI)
This repository includes the semester project of the course ATAI ([source repository](https://github.com/bauerDOTuzh/atai_project/tree/6869d96a0887d26fcd2b6cfd53ec756a54577295)). The code is written in `Python v3.12`.

Main packages: `speakeasypy`, `spacy`, `sentence_transformers`, `nltk`, `ollama`, `FastAPI`, `rdflib`, `rapidfuzz`, `torch`, `fuzzywuzzy`, `transformers`, `numpy`, `asyncio`, `logging`, `pandas`, `json`, `regex`

## How To Run
Login on Speakeasy: https://speakeasy.ifi.uzh.ch/

Request chat with: `fierce-blade`

## Local steps:
- change your shell to your environment (conda, venv, etc)
- `poetry install` to install dependencies and create environment
- `uvicorn app.main:app --reload` to run the project in DEV mode
- `uvicorn app.main:app` to run the project in PROD mode

## How To Download The Dataset
- run `python ./utils/download_dataset.py` to download the [dataset](https://files.ifi.uzh.ch/ddis/teaching/ATAI2024/dataset/)

# NLP Extractor Approach
- We have an NLP query
- We remove the question mark
1) Movie Extractor
    - Extract with NLTK with levhenstein distance for most similar words (with hypens variations if they exist)
    - If no match was found, we extract with fuzzy matcher (maximally one movie will be returned on this point right now)
    - it is theoretically possible to run NER on movies and then fuzzy match them against the extracted movie names but too much work
    - it would work good when we want to extract all movies from a text but nltk with better algo could solve this as well
2) Relationship extractor
    - Manual extraction (like publication date which is weird)
    - Extract with NLTK with levhenstein distance for most similar words (works perfectly for directors and similar)
    - Extract with fuzzy matcher (maximally one movie will be returned on this point right now - but not that good for relations)
    - Run embedding search
    - Take all relations and embedd them into space
    - take the query without movie remove all stopwords and embedd it, hopefully we get some relationships

# 2nd Evaulation - Example Queries
- Who directed the movie "The Matrix"?
- What is the genre of the movie "The Matrix"?
- What is the publication date of the movie "The Matrix"?
- What is the duration of the movie "The Matrix"?

# 3rd Evaluation - Example Queries
- "Given that I like Kung Fu Panda, can you recommend some movies?"
   - Have you considered watching Spy Game? I recommend it!
- "Recommend movies like Nightmare on Elm Street, Friday the 13th, and Halloween."
   - Have you considered watching C.H.U.D.? I recommend it!
- "Given that I like The Lion King, Pocahontas, and The Beauty and the Beast, can you recommend some movies?"
   - You might really like Sleeping Beauty, give it a watch!
- "Recommend some movies like The Avengers."
   - How about watching C.H.U.D.? It's worth your time.
- "Recommend movies like Inception."
   - There is no movie worthy of Inception to be recommended.
- "Recommend movies like Interstellar?"
   - Have you considered watching C.H.U.D.? I recommend it!
- "Recommend movies like The Godfathera."
   - You might really like Buffalo '66, give it a watch!
- "Recommend me a movie similar to Deadpool?"
   - Have you considered watching Day of the Comet? I recommend it!
- "Recommend me movie like Harry Potter."
   - I could not find any entity in your message, please try again and reformulate.
- "Recommend movies similar to Hamlet and Othello."
   - How about watching King Lear? It's worth your time.
- "Recommend movies like The Godfather."
   - I think you'd enjoy watching Buffalo '66.
- "I really like The Lion King and Pocahontas, can you recommend some similar movies?"
   - I think you'd enjoy watching Oliver & Company.
- "Recommend me a movie like Harry Potter."
   - I think you'd enjoy watching Forever Enthralled.
- "Recommend movies like Forest Gump and The Lord of the Rings: The Fellowship of the Ring."
   - I think you'd enjoy watching North.
- "Recommend some movies given that I like Ryan Gosling."
   - Have you considered watching Bukowski: Born into This? I recommend it!
- "Given that I like A.I. Artificial Intelligence, can you recommend some movies?"
   - You can't go wrong with Buffalo Girls, I highly recommend it.
- "If I like Matthew McConaughey, recommend some other actors I might like."
   - You can't go wrong with Theeran Athikaram Ondru, I highly recommend it.
- "I like wormhole, what movie would I like."
   - I could not find any entity in your message, please try again and reformulate.
