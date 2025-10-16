from src.metrics.utils import calc_cer, calc_wer


def main(path_to_file):
    with open(path_to_file, "r") as f:
        total_cer, total_wer = 0, 0
        for line in f.readlines():
            id, text, pred_text = line.split()
            total_cer += calc_cer(text, pred_text)
            total_wer += calc_wer(text, pred_text)
        n = len(f.readlines())
        print(f"CER: {total_cer / n}\nWER: {total_wer / n}")


if __name__ == "__main__":
    main()
