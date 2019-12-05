from armetrics import scorer
from armetrics.models import Segment


def test_segment_basic_score():
    "Test TP, TN, FP, FN assignation by comparing y_true and y_pred"
    assert scorer.segment_basic_score(True, True) == "TP"
    assert scorer.segment_basic_score(False, False) == "TN"
    assert scorer.segment_basic_score(False, True) == "FP"
    assert scorer.segment_basic_score(True, False) == "FN"

test_segment_basic_score()

def test_score_segments():

    Segment(1, 2, "TP")
