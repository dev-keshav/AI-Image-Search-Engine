"""
query_parser.py

Parses a natural language prompt into structured search conditions.
Extracts objects, counts, spatial positions, and actions.

This is a rule-based parser. For production, you could replace this
with an LLM-based parser.
"""

import re


# Known spatial keywords mapped to image regions
SPATIAL_KEYWORDS = {
    "left": "left",
    "left side": "left",
    "right": "right",
    "right side": "right",
    "top": "top",
    "upper": "top",
    "bottom": "bottom",
    "lower": "bottom",
    "center": "center",
    "middle": "center",
    "sky": "top",
    "ground": "bottom",
    "foreground": "bottom",
    "background": "top",
}

# Known action keywords
ACTION_KEYWORDS = [
    "flying", "running", "sitting", "standing", "walking",
    "swimming", "jumping", "sleeping", "eating", "playing",
    "driving", "riding", "smiling", "crying", "dancing",
]

# Number words to digits
NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "a": 1, "an": 1, "single": 1,
}


def parse_query(query):
    """
    Parse a natural language query into structured conditions.

    Input:
        "a bird flying in the sky and three trees on the left side"

    Output:
        {
            "raw_query": "...",
            "objects": [
                {
                    "name": "bird",
                    "count": 1,
                    "action": "flying",
                    "location": "top"
                },
                {
                    "name": "tree",
                    "count": 3,
                    "action": None,
                    "location": "left"
                }
            ]
        }
    """
    query_lower = query.lower().strip()

    # Split by "and" or commas to get individual object phrases
    # Also handle "with" as a separator
    phrases = re.split(r'\band\b|,|\bwith\b', query_lower)
    phrases = [p.strip() for p in phrases if p.strip()]

    objects = []

    for phrase in phrases:
        obj = _parse_phrase(phrase)
        if obj and obj["name"]:
            objects.append(obj)

    # If no objects were found, try treating the whole query as one phrase
    if not objects:
        obj = _parse_phrase(query_lower)
        if obj and obj["name"]:
            objects.append(obj)

    return {
        "raw_query": query,
        "objects": objects
    }


def _parse_phrase(phrase):
    """
    Parse a single phrase to extract object name, count, action, and location.
    """
    count = None
    action = None
    location = None

    # Extract count
    # Check for digit numbers first
    digit_match = re.search(r'(\d+)', phrase)
    if digit_match:
        count = int(digit_match.group(1))

    # Check for word numbers
    if count is None:
        for word, num in NUMBER_WORDS.items():
            # Match whole word only
            pattern = r'\b' + re.escape(word) + r'\b'
            if re.search(pattern, phrase):
                count = num
                break

    # Default count
    if count is None:
        count = 1

    # Extract action
    for act in ACTION_KEYWORDS:
        if act in phrase:
            action = act
            break

    # Extract location
    for keyword, loc in SPATIAL_KEYWORDS.items():
        if keyword in phrase:
            location = loc
            break

    # Extract object name
    # Remove numbers, actions, spatial words, articles, prepositions
    name = phrase

    # Remove number words and digits
    for word in NUMBER_WORDS.keys():
        name = re.sub(r'\b' + re.escape(word) + r'\b', '', name)
    name = re.sub(r'\d+', '', name)

    # Remove action words
    for act in ACTION_KEYWORDS:
        name = name.replace(act, '')

    # Remove spatial keywords
    for keyword in SPATIAL_KEYWORDS.keys():
        name = name.replace(keyword, '')

    # Remove common filler words
    fillers = [
        'the', 'in', 'on', 'at', 'of', 'is', 'are', 'there',
        'only', 'just', 'some', 'from', 'to', 'with', 'that',
        'which', 'where', 'give', 'me', 'picture', 'photo',
        'image', 'show', 'find', 'search', 'get',
    ]
    for filler in fillers:
        name = re.sub(r'\b' + re.escape(filler) + r'\b', '', name)

    # Clean up whitespace
    name = re.sub(r'\s+', ' ', name).strip()

    # Remove trailing/leading punctuation
    name = name.strip('.,!? ')

    return {
        "name": name if name else None,
        "count": count,
        "action": action,
        "location": location,
    }


def get_detection_labels(parsed_query):
    """
    Convert parsed query objects into Grounding DINO detection labels.
    Grounding DINO expects labels separated by periods.

    Example:
        ["bird", "tree"] -> "bird. tree."
    """
    labels = []
    for obj in parsed_query["objects"]:
        if obj["name"]:
            labels.append(obj["name"])

    return labels


def get_vqa_questions(parsed_query):
    """
    Generate verification questions from parsed query.

    Example output:
        [
            "Is there a bird flying in the sky?",
            "How many trees are on the left side?",
        ]
    """
    questions = []

    for obj in parsed_query["objects"]:
        name = obj["name"]
        if not name:
            continue

        count = obj["count"]
        action = obj["action"]
        location = obj["location"]

        # Question about existence
        if action:
            questions.append(f"Is there a {name} {action}?")
        else:
            questions.append(f"Is there a {name} in this image?")

        # Question about count
        if count and count > 1:
            questions.append(f"How many {name}s are there?")

        # Question about location
        if location:
            questions.append(f"Is the {name} on the {location}?")

    return questions