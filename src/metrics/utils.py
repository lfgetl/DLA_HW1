# Based on seminar materials

# Don't forget to support cases when target_text == ''
import editdistance


def calc_wer(target_text: str, predicted_text: str) -> float:
    pred_array = []
    if predicted_text is not None:
        pred_array = predicted_text.split()

    return editdistance.eval(target_text.split(), pred_array) / len(target_text.split())


def calc_cer(target_text: str, predicted_text: str) -> float:
    if target_text is None:
        if predicted_text is None:
            return 0.0
        return len(predicted_text)
    if predicted_text is None:
        return len(target_text)
    return editdistance.eval(target_text, predicted_text) / len(target_text)
