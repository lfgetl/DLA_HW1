import re
from collections import defaultdict
from string import ascii_lowercase

import torch
from torchaudio.models.decoder import ctc_decoder, download_pretrained_files

# TODO add CTC decode
# TODO add BPE, LM, Beam Search support
# Note: think about metrics and encoder
# The design can be remarkably improved
# to calculate stuff more efficiently and prettier


class CTCTextEncoder:
    EMPTY_TOK = ""

    def __init__(self, alphabet=None, lm_name=None, beam_size=None, **kwargs):
        """
        Args:
            alphabet (list): alphabet for language. If None, it will be
                set to ascii
        """
        self.beam_size = beam_size
        self.lm_name = lm_name
        if self.lm_name is not None:
            files = download_pretrained_files(lm_name)
            self.decoder = ctc_decoder(
                lexicon=files.lexicon,
                tokens=files.tokens,
                lm=files.lm,
                nbest=1,
                beam_size=beam_size,
            )
            with open(files.tokens, "r") as file:
                self.vocab = [s.strip() for s in file.readlines()]
        else:
            if alphabet is None:
                self.alphabet = list(ascii_lowercase + " ")
            self.vocab = [self.EMPTY_TOK] + list(self.alphabet)

        self.ind2char = dict(enumerate(self.vocab))
        self.char2ind = {v: k for k, v in self.ind2char.items()}
        if self.lm_name is not None:
            self.char2ind[" "] = self.char2ind["|"]
            self.ind2char[self.char2ind[" "]] = " "
            self.ind2char[self.char2ind["-"]] = ""

    def __len__(self):
        return len(self.vocab)

    def __getitem__(self, item: int):
        assert type(item) is int
        return self.ind2char[item]

    def encode(self, text) -> torch.Tensor:
        text = self.normalize_text(text)
        try:
            return torch.Tensor([self.char2ind[char] for char in text]).unsqueeze(0)
        except KeyError:
            unknown_chars = set([char for char in text if char not in self.char2ind])
            raise Exception(
                f"Can't encode text '{text}'. Unknown chars: '{' '.join(unknown_chars)}'"
            )

    def decode(self, probs) -> str:
        """
        Raw decoding without CTC.
        Used to validate the CTC decoding implementation.

        Args:
            inds (list): list of tokens.
        Returns:
            raw_text (str): raw text with empty tokens and repetitions.
        """
        if self.lm_name is None:
            inds = probs.cpu().argmax(-1).numpy()
            return "".join([self.ind2char[int(ind)] for ind in inds]).strip()
        else:
            return self.ctc_decode(probs)

    def ctc_decode(self, probs) -> str:
        if self.lm_name is not None:
            return " ".join(self.decoder(probs.unsqueeze(0).exp().cpu())[0][0].words)
        inds = probs.cpu().argmax(-1).numpy()
        prev_ind = None
        res = ""
        for ind in inds:
            cur_ind = ind
            if cur_ind == prev_ind:
                continue
            prev_ind = cur_ind
            res += self.ind2char[cur_ind]

        return res

    @staticmethod
    def normalize_text(text: str):
        text = text.lower()
        text = re.sub(r"[^a-z ]", "", text)
        return text


def expand_and_merge_beams(dp, cur_step_prob, ind2char, EMPTY_TOK, beam_size):
    new_dp = defaultdict(float)
    topk_vals, topk_inds = torch.topk(cur_step_prob, k=beam_size, dim=-1)
    for (pref, prev_char), pref_proba in dp.items():
        for idx, val in zip(topk_inds.tolist(), topk_vals.tolist()):
            if idx not in ind2char:
                continue
            char = ind2char[idx]
            cur_proba = pref_proba + val
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
    srt = sorted(list(dp.items()), key=lambda x: -x[1])[:beam_size]
    return {tup: prob for tup, prob in srt}


def ctc_beam_search(probs, beam_size, ind2char, EMPTY_TOK):
    dp = {
        ("", EMPTY_TOK): 0.0,
    }

    for cur_step_prob in probs:
        dp = expand_and_merge_beams(dp, cur_step_prob, ind2char, EMPTY_TOK, beam_size)
        dp = truncate_beams(dp, beam_size)
    return dp


class BeamSearchEncoder:
    EMPTY_TOK = ""

    def __init__(self, alphabet=None, beam_size=None, **kwargs):
        """
        Args:
            alphabet (list): alphabet for language. If None, it will be
                set to ascii
        """
        self.beam_size = beam_size

        if alphabet is None:
            alphabet = list(ascii_lowercase + " ")

        self.alphabet = alphabet
        self.vocab = [self.EMPTY_TOK] + list(self.alphabet)

        self.ind2char = dict(enumerate(self.vocab))
        self.char2ind = {v: k for k, v in self.ind2char.items()}

    def __len__(self):
        return len(self.vocab)

    def __getitem__(self, item: int):
        assert type(item) is int
        return self.ind2char[item]

    def encode(self, text) -> torch.Tensor:
        text = self.normalize_text(text)
        try:
            return torch.Tensor([self.char2ind[char] for char in text]).unsqueeze(0)
        except KeyError:
            unknown_chars = set([char for char in text if char not in self.char2ind])
            raise Exception(
                f"Can't encode text '{text}'. Unknown chars: '{' '.join(unknown_chars)}'"
            )

    def decode(self, probs) -> str:
        """
        Raw decoding is the same as beam search
        """
        return self.ctc_decode(probs)

    def ctc_decode(self, probs) -> str:
        return sorted(
            list(
                ctc_beam_search(
                    probs, self.beam_size, self.ind2char, EMPTY_TOK=""
                ).items()
            )
        )[-1][0][0]

    @staticmethod
    def normalize_text(text: str):
        text = text.lower()
        text = re.sub(r"[^a-z ]", "", text)
        return text
