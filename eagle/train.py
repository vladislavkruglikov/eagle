import os
import json
import torch
import pathlib
import argparse
import accelerate
import safetensors
import transformers

from eagle.dataset import Dataset
from eagle.collator import Collator
from eagle.model import Model


def _train() -> None:
    arguments = _parse_arguments()
    
    train_dataset_path: pathlib.Path = arguments.train_input
    test_dataset_path: pathlib.Path = arguments.test_input
    model_path: pathlib.Path = arguments.model
    device: str = arguments.device
    max_model_len = arguments.max_model_len
    epochs = arguments.epochs
    v_w = 1.0
    p_w = 0.1
    grad_clip = 0.5
    save_freq = arguments.save_freq
    cpdir = arguments.cpdir
    eagle_config_path = arguments.eagle_config
    lr = arguments.lr

    print("Initializing accelerate")
    torch.backends.cuda.matmul.allow_tf32 = True
    accelerate.utils.set_seed(0)
    accelerator = accelerate.Accelerator()
    
    if accelerator.is_main_process:
        print("Initializing wandb")
        import wandb
        wandb.init(project="eagle")

    print("Initializing lm head")
    lm_head = _initialize_verifier_lm_head(verifier_path=model_path, device=device)
    # lm_head = lm_head.to(torch.float32)
    # print(next(lm_head.parameters()).dtype)

    print("Initializing datasets")
    train_dataset = Dataset(dataset_path=train_dataset_path, max_model_len=max_model_len)
    test_dataset = Dataset(dataset_path=test_dataset_path, max_model_len=max_model_len)
    train_data_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=1, num_workers=0, collate_fn=Collator())
    test_data_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1, num_workers=0, collate_fn=Collator())
    
    print("Initializing eagle model")
    config = transformers.AutoConfig.from_pretrained(eagle_config_path)
    model = Model(config, load_emb=True, path=model_path)#.to(config.torch_dtype)
    # for k, v in model.named_parameters():
        # print(k, v.dtype)

    print("Initializing loss, optimizers, etc")
    criterion = torch.nn.SmoothL1Loss(reduction="none")
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95))

    model, lm_head, optimizer, train_data_loader, test_data_loader = accelerator.prepare(
        model, lm_head, optimizer, train_data_loader, test_data_loader
    )

    model.train()

    print("Starting training loop")
    for epoch in range(epochs):
        num_batches = 0
        epoch_sum_loss = 0.0
        for batch in train_data_loader:
            with accelerator.accumulate(model):
                optimizer.zero_grad()
                predict = model(batch["hidden_states"], input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])

                with torch.no_grad():
                    target_head = lm_head(batch["target"])
                    target_p = torch.nn.Softmax(dim=2)(target_head)
                    target_p = target_p.detach()
                
                out_head = lm_head(predict)
                out_logp = torch.nn.LogSoftmax(dim=2)(out_head)

                loss_mask = batch["loss_mask"][:, :, None]
                plogp = target_p * out_logp
                ploss = -torch.sum(torch.sum(loss_mask * plogp, 2)) / (loss_mask.sum()+1e-5)

                vloss = criterion(predict, batch["target"])
                vloss = torch.sum(torch.mean(loss_mask * vloss, 2)) / (loss_mask.sum()+1e-5)

                loss = v_w * vloss + p_w * ploss
                accelerator.backward(loss)
                accelerator.clip_grad_value_(model.parameters(), grad_clip)

                num_batches += 1
                epoch_sum_loss += loss.item()

                optimizer.step()
        
        epoch_mean_loss = epoch_sum_loss / num_batches

        if accelerator.is_local_main_process:
            print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, epochs, epoch_mean_loss))
            wandb.log({"train/epochloss": epoch_mean_loss})
            if (epoch + 1) % save_freq == 0:
                _save_vllm_checkpoint(
                    accelerator=accelerator,
                    model=model,
                    eagle_config_path=eagle_config_path,
                    save_directory=f"{cpdir}/epoch_{epoch + 1}"
                )


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Process a raw chat dataset using a verifier model and tokenizer."
    )
    parser.add_argument(
        "--train-input",
        type=pathlib.Path,
        required=True,
        help="Path to tokenized train dataset folder"
    )
    parser.add_argument(
        "--test-input",
        type=pathlib.Path,
        required=True,
        help="Path to tokenized test dataset folder"
    )
    parser.add_argument(
        "--model",
        type=pathlib.Path,
        required=True,
        help="Path to verifier model"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device that will be used by the large model to generate hidden states (e.g., 'cpu', 'cuda:0')"
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=2048,
        help="Max model len"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="epochs"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning reate"
    )
    parser.add_argument(
        "--cpdir",
        type=pathlib.Path,
        default="./checkpoints",
        help="path to folder to save vllm checkpoints"
    )
    parser.add_argument(
        "--save_freq",
        type=int,
        default=10,
        help="save_freq"
    )
    parser.add_argument(
        "--eagle-config",
        type=pathlib.Path,
        help="path to eagle config"
    )
    return parser.parse_args()


def _initialize_verifier_lm_head(verifier_path: pathlib.Path, device: str) -> torch.nn.Linear:
    with open(f"{verifier_path}/config.json", "r") as file:
        config = json.load(file)
    head = torch.nn.Linear(config["hidden_size"], config["vocab_size"], bias=False)
    with open(os.path.join(verifier_path, "model.safetensors.index.json"), "r") as f:
        index_json = json.loads(f.read())
    head_path = index_json["weight_map"]["lm_head.weight"]
    with safetensors.safe_open(os.path.join(verifier_path, head_path), framework="pt", device=device) as f:
        tensor = f.get_slice("lm_head.weight")[:, :config["hidden_size"]]
    head.weight.data = tensor
    head.eval()
    for param in head.parameters():
        param.requires_grad = False
    return head


def _save_vllm_checkpoint(
    accelerator: accelerate.Accelerator, 
    model: torch.nn.Module, 
    eagle_config_path: pathlib.Path, 
    save_directory: pathlib.Path
) -> None:
    accelerator.save_model(model, save_directory=save_directory)
    with open(eagle_config_path) as rf:
        cfg = json.load(rf)
    cfg = {"model_type": "eagle", "model": cfg}
    with open(f"{save_directory}/config.json", "w") as wf:
        json.dump(cfg, wf, indent=4)


if __name__ == "__main__":
    _train()
