# Based on seminar materials

# Don't forget to support cases when target_text == ''
from collections import defaultdict

import editdistance


def calc_wer(target_text: str, predicted_text: str) -> float:
    pred_array = []
    if predicted_text is not None:
        pred_array = predicted_text.lower().split()
    if target_text is None or len(target_text.split()) == 0:
        return len(pred_array)
    return editdistance.eval(target_text.lower().split(), pred_array) / len(
        target_text.lower().split()
    )


def calc_cer(target_text: str, predicted_text: str) -> float:
    if target_text is None or len(target_text) == 0:
        if predicted_text is None:
            return 0.0
        return len(predicted_text)
    if predicted_text is None:
        return len(target_text)
    return editdistance.eval(target_text.lower(), predicted_text.lower()) / len(
        target_text.lower()
    )
