import torch
from torch.nn.utils.rnn import pad_sequence


def collate_fn(dataset_items: list[dict]):
    """
    Collate and pad fields in the dataset items.
    Converts individual items into a batch.

    Args:
        dataset_items (list[dict]): list of objects from
            dataset.__getitem__.
    Returns:
        result_batch (dict[Tensor]): dict, containing batch-version
            of the tensors.
    """
    result_batch = dict()
    for k in dataset_items[0].keys():
        result_batch[k] = []
    result_batch["spectrogram_length"] = []
    result_batch["text_encoded_length"] = []
    for el in dataset_items:
        for k, v in el.items():
            if k in result_batch.keys():
                if k in ["audio", "text", "audio_path"]:
                    result_batch[k].append(v)
                else:
                    dv = v.squeeze(0)
                    if k == "spectrogram":
                        dv = dv.permute(1, 0)
                    result_batch[k + "_length"].append(dv.size(dim=0))
                    result_batch[k].append(dv)

    result_batch["spectrogram"] = pad_sequence(
        result_batch["spectrogram"], batch_first=True
    )
    result_batch["spectrogram"] = result_batch["spectrogram"].permute(0, 2, 1)
    result_batch["text_encoded"] = torch.nn.utils.rnn.pad_sequence(
        [elem["text_encoded"].squeeze(0) for elem in dataset_items], batch_first=True
    )
    result_batch["spectrogram_length"] = torch.Tensor(
        result_batch["spectrogram_length"]
    ).long()
    result_batch["text_encoded_length"] = torch.Tensor(
        result_batch["text_encoded_length"]
    ).long()

    return result_batch
