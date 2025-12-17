from utils.helper import helper
from models.sentiment import EmotionClassifier


def test_deduplicate_texts_removes_duplicates_and_short_texts():
    texts = ["Hello world", "hello   world ", "short", "another long text"]
    unique = helper.deduplicate_texts(texts, min_length=10)
    # "Hello world" and "hello   world " should collapse into one
    assert len(unique) == 2
    assert "short" not in unique


def test_emotion_classifier_keywords():
    clf = EmotionClassifier()
    assert clf.predict("Markets crash, panic everywhere") == "fear"
    assert clf.predict("Greed drives this rally to the moon") == "greed"
    assert clf.predict("Sideways consolidation today") == "neutral"


