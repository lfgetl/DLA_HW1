import sys

from src.metrics.utils import calc_cer, calc_wer


def calc_metric(path):
    with open(path, "r") as f:
        total_cer, total_wer = 0, 0
        n = 0
        for line in f.readlines():
            n += 1
            line = line.strip()
            id, text, pred_text = line.split("\t")
            total_cer += calc_cer(text, pred_text)
            total_wer += calc_wer(text, pred_text)
        print(f"CER: {total_cer / n}\nWER: {total_wer / n}")


if __name__ == "__main__":
    calc_metric(sys.argv[1])
