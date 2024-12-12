import os
import json
import spacy
from app.config.app import settings
import re
from rapidfuzz import process, fuzz
import regex

class Extractor:
    def __init__(self):
        self.map = {}
        with open(os.sep.join((settings.utils_path, "useful_dataset/graph/unique_movies.json"))) as f:
            self.entities = json.load(f)
            # use entities map from lowercase to entity uppercase
            self.map['movie'] = {entity.lower(): entity for entity in self.entities}

        with open(os.sep.join((settings.utils_path, "useful_dataset", "graph", "unique_relationships.json"))) as f:
            self.relations = json.load(f)
            self.map['relation'] = {relation.lower(): relation for relation in self.relations}

        with open(os.sep.join((settings.utils_path, "useful_dataset", "graph", "unique_persons.json"))) as f:
            self.persons = json.load(f)
            self.map["person"] = {person.lower(): person for person in self.persons}

class SpacyExtractor(Extractor):
    def __init__(self):
        # Load the spaCy model from the saved path
        path = os.sep.join((
            settings.utils_path,
            "test_scripts",
            "spacy_rule_based_recognition",
            "movie_relation_ner_model"
        ))
        self.nlp = spacy.load(path)
        self.score_threshold_with_nltk = 90  # Threshold with NLTK matches
        self.score_threshold_without_nltk = 80  # Threshold without NLTK matches
        self.hyphen_variants = ['-', '–', '—']
        super().__init__()  # Get relations and entities from the parent class

    def preprocess_text(self, text: str) -> str:
        # Standardize text: lowercase and handle colons
        text = text.lower()
        text = re.sub(r":", ":", text)  # This line seems redundant but kept for consistency
        return text

    def generate_hyphen_variants(self, text: str) -> list:
        """
        Generate all mutated versions of the input text by replacing hyphens with different variants.
        """
        hyphen_pattern = r'[-–—]'
        if not re.search(hyphen_pattern, text):
            return [text]  # No hyphens to replace

        # Generate mutated texts by replacing all hyphens with each variant
        mutated_texts = []
        for variant in self.hyphen_variants:
            mutated = re.sub(hyphen_pattern, variant, text)
            mutated_texts.append(mutated)
        return mutated_texts

    def get_entities_with_fuzzy_matching(self, text: str, type: str) -> dict:
        if type not in ['movie', 'relation', 'person']:
            raise ValueError(f"Invalid type '{type}' specified. Must be one of 'movie', 'relation', 'person', or 'all'.")
        
        text = self.preprocess_text(text)  # Preprocess the text for consistent matching

        # Generate all hyphen variants
        mutated_texts = self.generate_hyphen_variants(text)

        nltk_matches_all_variants = []
        fuzzy_matches_all_variants = []

        for mutated_text in mutated_texts:
            print(f"Mutated text: {mutated_text}")
            # Get NLTK matches for the mutated text
            nltk_matches = self.get_nltk_matches(mutated_text, type)
            nltk_matches_all_variants.extend(nltk_matches)
            print(f"NLTK matches: {nltk_matches}")

            # Get fuzzy matches for the mutated text
            score = self.score_threshold_with_nltk if nltk_matches else self.score_threshold_without_nltk
            fuzzy_matches = self.get_fuzzy_matches(mutated_text, type, default_score=score)
            fuzzy_matches_all_variants.extend(fuzzy_matches)
            
            print(f"Fuzzy matches: {fuzzy_matches}")

        # Combine matches from all variants
        combined_results = self.combine_and_sort_matches(
            nltk_matches_all_variants, fuzzy_matches_all_variants
        )

        # filter out weird words
        # if there are more than 1 results and one of them is "tell" remove it
        print(f"Combined results: {combined_results}")
        if len(combined_results) > 1:
            combined_results = [result for result in combined_results if result['original_text'] != 'tell']

        

        # transform all results into their uppercase version -> then they are only array
        combined_results = self.convert_to_correct_name(combined_results, type)

        # Return results grouped by type
        return {
            "movie": combined_results if type == "movie" else [],
            "relation": combined_results if type == "relation" else [],
            "person": combined_results if type == "person" else [],
        }
    
    # convert to correct uppernamed name, everything now was done on lowercase
    def convert_to_correct_name(self, matches, type):

        return [self.map[type][match['original_text'].lower()] for match in matches]

    def get_nltk_matches(self, text, type):
        """
        Extract entities using the spaCy model (NLTK-like behavior).
        """
        doc = self.nlp(text)
        matches = []

        for ent in doc.ents:
            ent_label = ent.label_
            if type in [ent_label]:
                # filter out word tell if it is in the first 1/3 of the sentence
                # if ent.text == "tell":
                    # if ent.start_char < (len(text) / 5):
                        # print(f"Tell is not full word, because {ent.start_char} is smaller than {len(text) / 3}")
                        # continue

                matches.append({'original_text': ent.text, 'start': ent.start_char, 'end': ent.end_char})

        # remove duplicates
        matches = [dict(t) for t in {tuple(d.items()) for d in matches}]

        matches = self.filter_overlapping_matches(matches)

        print(f"Matches after filtering overlapping matches: {matches}")

        return matches
    
    def spans_overlap(self, start1, end1, start2, end2):
        """
        Check if two spans overlap.
        """
        return max(start1, start2) < min(end1, end2)
    
    def filter_overlapping_matches(self, matches):
        """
        Remove overlapping matches, keeping only the best one based on score, length, and position.
        """

        filtered_matches = []
        for match in matches:
            start = match["start"]
            end = match["end"]

            # Check for overlap with already kept matches
            overlaps = any(self.spans_overlap(start, end, kept_match["start"], kept_match["end"])
                        for kept_match in filtered_matches)

            if not overlaps:
                filtered_matches.append(match)

        return filtered_matches
    

    def get_fuzzy_matches(self, text, type, default_score:int = 90 ):
        """
        Extract entities using fuzzy matching.
        """
        text += " "  # Add a space at the end to ensure full word matches - works better 
        matches = []
        candidates = list(self.map[type].keys())

        threshold = default_score

        matches = []

        alignment_result = fuzz.partial_ratio_alignment(
                candidates, text, processor=None, score_cutoff=threshold
        )

        print(f"Alignment result: {alignment_result}")

        for candidate in candidates:
            # Compute alignment for each candidate
            alignment_result = fuzz.partial_ratio_alignment(
                candidate, text, processor=None, score_cutoff=threshold
            )

            if alignment_result is None:
                continue  # Skip if no match found

            score = alignment_result.score
            # src_start = alignment_result.src_start
            # src_end = alignment_result.src_end
            dest_start = alignment_result.dest_start
            dest_end = alignment_result.dest_end

            # for short words take long threshold, ignoring the etc not possible
            curr_threshold = threshold if len(candidate) >= 5 else 95
            
            if score >= curr_threshold:
                # Extract the matching substring from the input text
                matching_substring = text[dest_start:dest_end].strip()
                # remove " " " from start of the string if there
                if matching_substring[0] == " \"":
                    matching_substring = matching_substring[1:]

                # Check if the match is a full word (or space in the end etc)
                is_full_word = (
                    (dest_start == 0 or text[dest_start - 1] in {' ', "'", '"',".",","})  # Start or preceded by space, apostrophe, or quote
                    and (dest_end == len(text) or text[dest_end] in {' ', "'", '"',".", ","})  # End or followed by space, apostrophe, or quote
                )
                # print(f"Matches for candidate '{candidate}': {alignment_result}")

                # Second fallback - it is most probably a full word if we take some chars from prev
                # "Nightmare on Elm Street" is -> "A Nightmare on Elm Street"
                # dest_start is "a"
                # print(f"Deciding on text_start is: {text[dest_start]} and text_end is: {text[dest_end] if dest_end < len(text) else 'None'}")
                
                if not is_full_word:
                    # we assume that the word must have at least 3 chars to be considered in this fallback scenario
                    if len(matching_substring) > 3:
                        # we assume that 
                        if dest_start > 0:
                            # it is space - we detected the word with a space, which is interesant but valid
                            if matching_substring == " ":
                                # make if a full word if it ends with something at the end
                                is_full_word = True if dest_end == len(text) or text[dest_end] in {' ', "'", '"',".", ","} else False
                                # print(f"is is full word because of last entity")
                            else:
                                # assume that matching string is a character - like "A" from "A Nightmare on Elm Street"
                                if candidate.startswith('a') or candidate.startswith('the'):
                                    is_full_word = True if dest_end == len(text) or text[dest_end] in {' ', "'", '"',".", ","} else False
                                    # print(f"is is full word because of a and the")

                # filter out the word tell
                # if candidate == "tell":
                #     print(f"Deciding on tell text_start is: {text[dest_start]} and text_end is: {text[dest_end] if dest_end < len(text) else 'None'}")
                #     # print(text, len(text), dest_start, len(text) / 3)
                #     print(f"Text: {text}, dest_start: {dest_start}, len(text) / 3: {len(text) / 3}")
                #     # check the position of tell in the text (if it appears in the first 1/5 of the complete sentence, then it is not a full word)
                #     if dest_start < (len(text) / 5):
                #         is_full_word = False
                #         print(f"Tell is not full word, because {dest_start} is smaller than {len(text) / 3}")
                #         continue
                #     else:
                #         print(f"Tell is full word, because {dest_start} is bigger than {len(text) / 3}")


                # Map candidate to its original text and append the result
                # original_text = self.map[type][candidate]
                matches.append({
                    "match_text": matching_substring,  # Substring from the input text
                    "original_text": candidate,    # Original candidate text
                    "label": self.get_candidate_type(candidate),
                    "score": score,
                    "start": dest_start,
                    "end": dest_end,
                    "is_full_word": is_full_word
                })
                
            # Uncomment this for a good debugging
            # print(f"Matches for candidate '{candidate}': {alignment_result}")

        # use only full words
        # print(f"Matches before filtering full word matches: {matches}")
        
        matches = [match for match in matches if match["is_full_word"]]
        print(f"Full word matches: {matches}")

        #filter out overlapping with the highest scores

        matches = self.filter_overlapping_matches(matches)
        # Sort matches by score descending and length descending
        matches = sorted(matches, key=lambda x: (-x["score"], -len(x["original_text"])))

        return matches
    
    def get_fuzzy_matches_full_words(self, text, type, default_score:int = 90 ):
        """
        Get best suitable entity for a given text
        """
        candidates = list(self.map[type].keys())
        filtered_candidates = []
    
        for candidate in candidates:
            # Use fuzz.partial_ratio to calculate similarity
            score = fuzz.ratio(text, candidate)
            if score >= default_score:
                # Add the score to the candidate dictionary
                result = {
                    "match_text": text,
                    "original_text": candidate,
                    "score": score
                }
                filtered_candidates.append(result)
        
        return filtered_candidates


    def get_candidate_type(self, candidate):
        """
        Determine the type of a candidate entity.
        """
        candidate_lower = candidate.lower()
        if candidate_lower in map(str.lower, self.entities):
            return "movie"
        elif candidate_lower in map(str.lower, self.relations):
            return "relation"
        elif candidate_lower in map(str.lower, self.persons):
            return "person"
        return "unknown"

    def combine_and_sort_matches(self, nltk_matches, fuzzy_matches):
        """
        Combine NLTK and fuzzy matches from all variants, prioritize longer matches
        even if they are from fuzzy matching, and ensure one match per place.
        """
        # Assign a default score of 100 to NLTK matches for consistent comparison
        for match in nltk_matches:
            match["score"] = 100

        # Combine matches
        combined_matches = nltk_matches + fuzzy_matches

        # Sort matches by start position (ascending), then by length (descending), and finally by score (descending)
        combined_matches = sorted(combined_matches, key=lambda x: (x["start"], -len(x["original_text"]), -x["score"]))

        filtered_matches = []
        for match in combined_matches:
            start = match["start"]
            end = match["end"]

            # Check for overlap with already kept matches
            overlaps = [
                kept_match for kept_match in filtered_matches
                if self.spans_overlap(start, end, kept_match["start"], kept_match["end"])
            ]

            if overlaps:
                # Replace the overlapping match if the current match is longer or has a higher score
                overlapping_match = overlaps[0]  # There should only be one overlapping match due to earlier sorting
                if len(match["original_text"]) > len(overlapping_match["original_text"]):
                    filtered_matches.remove(overlapping_match)
                    filtered_matches.append(match)
                elif len(match["original_text"]) == len(overlapping_match["original_text"]) and match["score"] > overlapping_match["score"]:
                    filtered_matches.remove(overlapping_match)
                    filtered_matches.append(match)
            else:
                filtered_matches.append(match)

        # Sort final matches by start position for readability
        filtered_matches = sorted(filtered_matches, key=lambda x: x["start"])

        return filtered_matches