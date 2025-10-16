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
    for k in ["text", "audio_path"]:
        result_batch[k] = [elem[k] for elem in dataset_items]
    result_batch["audio"] = [elem["audio"].squeeze(0) for elem in dataset_items]
    result_batch["old_audio"] = [elem["audio"].squeeze(0) for elem in dataset_items]
    result_batch["spectrogram"] = pad_sequence(
        [elem["spectrogram"].squeeze(0).permute(1, 0) for elem in dataset_items],
        batch_first=True,
    )
    result_batch["spectrogram"] = result_batch["spectrogram"].permute(0, 2, 1)
    result_batch["text_encoded"] = pad_sequence(
        [elem["text_encoded"].squeeze(0) for elem in dataset_items], batch_first=True
    )
    result_batch["spectrogram_length"] = torch.Tensor(
        [elem["spectrogram"].shape[-1] for elem in dataset_items]
    ).long()
    result_batch["text_encoded_length"] = torch.Tensor(
        [elem["text_encoded"].shape[-1] for elem in dataset_items]
    ).long()

    return result_batch
