import torch
import streaming


class Dataset(streaming.StreamingDataset):
    def __init__(
        self, 
        max_model_len: int, 
        transform_uniform_low: float, 
        transformer_uniform_high: float,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._max_model_len = max_model_len
        self._transform_uniform_low = transform_uniform_low
        self._transformer_uniform_high = transformer_uniform_high
    
    def __getitem__(self, index: int):
        data = super().__getitem__(index)
        data = {k: torch.from_numpy(v) for k, v in data.items()}
        data["hidden_state"] = data["hidden_state"][:self._max_model_len]
        data["hidden_state"] = self._apply_noise_to_hidden_state(hidden_state=data["hidden_state"])
        data["input_ids"] = data["input_ids"][:self._max_model_len]
        data["loss_mask"] = data["loss_mask"][:self._max_model_len]
        data["attention_mask"] = torch.ones(data["loss_mask"].shape[-1], dtype=torch.long)
        data["input_ids"] = torch.cat((data["input_ids"][1:], torch.tensor([0])), dim=0)
        # hidden_state already exists and it is input. Remember in eagle paper we
        # sent hidden state + next token. But learn to predict next hidden state aswell
        # as next token
        data["target"] = torch.cat((data["hidden_state"][1:], torch.zeros(1, data["hidden_state"].shape[1])), dim=0)
        return data

    def _apply_noise_to_hidden_state(self, hidden_state: torch.FloatTensor) -> None:
        noise = torch.rand_like(hidden_state) * (self._transformer_uniform_high - self._transform_uniform_low) + self._transform_uniform_low
        noisy_tensor = hidden_state + noise
        return noisy_tensor


# if __name__ == "__main__":
#     dataset = Dataset(
#         remote="s3://your_bucket_name/dataset",
#         shuffle=True,
#         shuffle_seed=0,
#         batch_size=4,
#         max_model_len=8,
#         transform_uniform_low=-0.1,
#         transformer_uniform_high=0.1
#     )

#     print(len(dataset))
