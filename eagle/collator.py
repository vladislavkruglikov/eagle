import torch


class Collator:
    def paddingtensor(self, intensors, N):
        intensors = intensors.unsqueeze(0)
        B, n, S = intensors.shape
        padding_tensor = torch.zeros(B, N - n, S)
        outtensors = torch.cat((intensors.to("mps"), padding_tensor.to("mps")), dim=1)
        return outtensors

    def paddingtensor2D(self, intensors, N):
        intensors = intensors.unsqueeze(0)
        B, n = intensors.shape
        padding_tensor = torch.zeros(B, N - n, dtype=intensors.dtype)
        outtensors = torch.cat((intensors, padding_tensor), dim=1)
        return outtensors
    
    def __call__(self, features: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        max_length = max(item['hidden_state'].shape[0] for item in features)
        batch_input_ids = torch.cat([self.paddingtensor2D(item['input_ids'], max_length) for item in features])
        batch_hidden_states = torch.cat([self.paddingtensor(item['hidden_state'], max_length) for item in features])
        batch_target = torch.cat([self.paddingtensor(item['target'], max_length) for item in features])
        batch_loss_mask = torch.tensor([item['loss_mask'].tolist() + [0] * (max_length - len(item['loss_mask'])) for item in features])
        batch_attention_mask = torch.tensor([item['attention_mask'].tolist() + [0] * (max_length - len(item['attention_mask'])) for item in features])
        return {
            "input_ids": batch_input_ids,
            "hidden_states": batch_hidden_states,
            "target": batch_target,
            "attention_mask": batch_attention_mask,
            "loss_mask": batch_loss_mask,
        }


if __name__ == "__main__":
    from eagle.dataset import Dataset
    dataset = Dataset(dataset_path="./tokenized_dataset", max_model_len=2048)
    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=1, collate_fn=Collator())
    for batch in dataloader:
        # print(batch)
        # print(batch["input_ids"].shape)
        ...
