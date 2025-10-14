# Based on seminar materials

# Don't forget to support cases when target_text == ''
from collections import defaultdict

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


def expand_and_merge_beams(dp, cur_step_prob, ind2char, EMPTY_TOK):
    new_dp = defaultdict(float)

    for (pref, prev_char), pref_proba in dp.items():
        for idx, char in enumerate(ind2char):
            cur_proba = pref_proba * cur_step_prob[idx]
            cur_char = char

            if char == EMPTY_TOK:
                cur_pref = pref
            else:
                if prev_char != char:
                    cur_pref = pref + char
                else:
                    cur_pref = pref
            new_dp[(cur_pref, cur_char)] += cur_proba
    return new_dp


def truncate_beams(dp, beam_size):
    return sorted(list(dp.items()), key=lambda x: -x[1])[:beam_size]


def ctc_beam_search(probs, beam_size, ind2char, EMPTY_TOK):
    dp = {
        ("", EMPTY_TOK): 1.0,
    }

    for cur_step_prob in probs:
        dp = expand_and_merge_beams(dp, cur_step_prob, ind2char, EMPTY_TOK)
        dp = truncate_beams(dp, beam_size)
    return dp
