import os
import torch
import pathlib


class Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path: pathlib.Path, max_model_len: int):
        super().__init__()
        self._dataset_path = dataset_path
        self._dataset_size = len(os.listdir(dataset_path))
        self._max_model_len = max_model_len

    def __len__(self) -> int:
        return self._dataset_size

    def __getitem__(self, index: int):
        data = torch.load(f"{self._dataset_path}/{index}.ckpt", map_location="cpu")
        data["hidden_state"] = data["hidden_state"][:self._max_model_len]
        data["input_ids"] = data["input_ids"][:self._max_model_len]
        data["loss_mask"] = data["loss_mask"][:self._max_model_len]
        data["attention_mask"] = torch.ones(self._max_model_len)
        data["input_ids"] = torch.cat((data["input_ids"][1:], torch.tensor([0])), dim=0)
        # hidden_state already exists and it is input. Remember in eagle paper we
        # sent hidden state + next token. But learn to predict next hidden state aswell
        # as next token
        data["target"] = torch.cat((data["hidden_state"][1:], torch.zeros(1, data["hidden_state"].shape[1])), dim=0)
        return data
        

if __name__ == "__main__":
    dataset = Dataset(dataset_path="./tokenized_dataset", max_model_len=1)
    print(dataset[0])
