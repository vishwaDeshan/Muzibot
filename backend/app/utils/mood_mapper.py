from enum import Enum

class MoodEnum(Enum):
    HAPPY = 1
    SAD = 2
    ANGRY = 3
    RELAXED = 4

# Function to convert number to string
def get_mood_string(mood_number: int) -> str:
    try:
        return MoodEnum(mood_number).name.lower()
    except ValueError:
        return "unknown"