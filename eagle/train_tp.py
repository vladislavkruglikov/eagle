import os
import time
import json
import math
import torch
import pathlib
import clearml
import logging
import datasets
import argparse
import safetensors
import transformers

from eagle.llama2 import Llama2Model


def coach() -> None:
    arguments = _parse_arguments()

    torch.backends.cuda.matmul.allow_tf32 = True
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(asctime)s [%(filename)s:%(lineno)d] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    logging.info("Start to prepare clearml ")
    clearml_task = clearml.Task.init(project_name=arguments.clearml_project, task_name=arguments.clearml_task, reuse_last_task_id=False, continue_last_task=False, output_uri=False, auto_connect_frameworks=False, auto_resource_monitoring=False)
    clearml_logger = clearml_task.get_logger()
    
    logging.info("Start to prepare language model head ")
    lm_head = _initialize_verifier_lm_head(verifier_path=arguments.verifier_model_path).to(getattr(torch, arguments.verifier_model_lm_head_dtype)).to("cuda")
    logging.info("Language model head has dtype %s", next(lm_head.parameters()).dtype)
    logging.info("Language model head has %f billion parameters", _count_parameters(model=lm_head) / 10 ** 9)
    clearml_logger.report_single_value(name="Language model head parameters billion", value=_count_parameters(model=lm_head) / 10 ** 9)

    logging.info("Start to prepare target model ")
    verifier_model = transformers.AutoModelForCausalLM.from_pretrained(arguments.verifier_model_path, device_map="auto", torch_dtype=getattr(torch, arguments.verifier_model_dtype), attn_implementation=arguments.attn)
    verifier_model = verifier_model.eval()
    logging.info("Target model head has dtype %s", next(verifier_model.parameters()).dtype)
    logging.info("Target model head has %f billion parameters", _count_parameters(model=verifier_model) / 10 ** 9)
    clearml_logger.report_single_value(name="Target model head parameters billion", value=_count_parameters(model=verifier_model) / 10 ** 9)

    logging.info("Start to prepare draft model ")
    config = transformers.AutoConfig.from_pretrained(arguments.eagle_config_path)
    model = Llama2Model(config, load_emb=True, path=arguments.verifier_model_path).to(getattr(torch, arguments.eagle_dtype)).to("cuda")
    logging.info("Draft model has dtype %s", next(model.parameters()).dtype)
    logging.info("Draft model has %f billion parameters", _count_parameters(model=model) / 10 ** 9)
    model.train()
    clearml_logger.report_single_value(name="Draft model head parameters billion", value=_count_parameters(model=model) / 10 ** 9)

    logging.info("Start to prepare data ")
    dataset = datasets.load_dataset("json", data_files={"train": [arguments.dataset_path]})["train"]
    dataset = Dataset(dataset=dataset)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=arguments.micro_batch_size, collate_fn=Collator(arguments.verifier_model_path))
    logging.info("Dataset contains %d samples", len(dataset))

    logging.info("Loading token mapping cache...")
    cache = torch.load(arguments.token_mapping_cache)
    d2t = cache["d2t"]
    t2d = cache["t2d"]
    draft_vocab_size = sum(t2d)
    logging.info("Vocab size of draft model is %d", draft_vocab_size)
    d2t = d2t.to("cuda")
    t2d = t2d.to("cuda")

    logging.info("Vocab size of draft model is %d", draft_vocab_size)


    logging.info("Start to prepare miscellaneous ")
    criterion = torch.nn.SmoothL1Loss(reduction="none")
    model_optimizer = torch.optim.AdamW(model.parameters(), lr=arguments.learning_rate, betas=(arguments.b1, arguments.b2))

    scheduler = transformers.get_linear_schedule_with_warmup(optimizer=model_optimizer, num_warmup_steps=arguments.num_warmup_steps, num_training_steps=arguments.num_training_steps)

    logging.info("Start training ")
    total_steps_passed = 0
    for epoch in range(arguments.epochs):
        training_iterator = iter(dataloader)
        num_samples_in_epoch = len(dataloader)
        remainder = num_samples_in_epoch % arguments.gradient_accumulation_steps
        remainder = remainder if remainder != 0 else arguments.gradient_accumulation_steps
        total_gradient_updates = math.ceil(num_samples_in_epoch / arguments.gradient_accumulation_steps)
        for update_step in range(total_gradient_updates):
            step_start = time.perf_counter()
            accum_loss = 0.0
            batch_samples = []
            num_batches_in_step = arguments.gradient_accumulation_steps if update_step != (total_gradient_updates - 1) else remainder
            for _ in range(num_batches_in_step):
                batch_samples += [next(training_iterator)]   
            num_items_in_batch = sum([batch["loss_mask"][:, :arguments.maximum_model_length].sum() for batch in batch_samples])

            if num_items_in_batch == 0:
                logging.warning("num_items_in_batch is zero, skipping steps")
                continue

            step_correctly_predicted_tokens_count = 0

            for i, batch in enumerate(batch_samples):
                batch = _make_eagle_input(batch, verifier_model, arguments.maximum_model_length, arguments.noise_low, arguments.noise_high, "cuda")
                batch["hidden_states"] = batch["hidden_states"]
                batch["target"] = batch["target"]
                predict = model(batch["hidden_states"].to( getattr(torch, arguments.eagle_dtype) ), input_ids=batch["input_ids"])

                loss_mask = batch["loss_mask"][:, :, None]

                with torch.no_grad():
                    target_head = lm_head(batch["target"].to( getattr(torch, arguments.verifier_model_lm_head_dtype) ),)

                    target_max_token = target_head.argmax(-1)
                    target_mask = t2d[target_max_token]
                    target_mask = target_mask[..., None].int()
                    position_mask = target_mask * loss_mask
                    target_head = target_head[..., t2d]
                    target_head = target_head.float()

                    target_p = torch.nn.Softmax(dim=2)(target_head)
                    target_p = target_p.detach()

                out_head = torch.nn.functional.linear(predict.to(getattr(torch, arguments.verifier_model_lm_head_dtype)), lm_head.weight[t2d, :], lm_head.bias)
                out_logp = torch.nn.LogSoftmax(dim=2)(out_head)

                
                _, target_max_p_tokens = torch.max(target_p, 2)
                _, ealge_max_p_tokens = torch.max(out_logp, 2)
                step_correctly_predicted_tokens_count += ((target_max_p_tokens == ealge_max_p_tokens) * loss_mask.squeeze()).sum().item()

                plogp = target_p * out_logp
                ploss = -torch.sum(torch.sum(position_mask * plogp, 2))
                vloss = criterion(predict, batch["target"])
                vloss = torch.sum(torch.mean(loss_mask * vloss, 2))
                loss = arguments.v_w * vloss + arguments.p_w * ploss
                loss = loss / num_items_in_batch
                accum_loss += loss.item()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), arguments.grad_clip)
            
            model_optimizer.step()
            scheduler.step()
            model_optimizer.zero_grad()
            
            total_steps_passed += 1

            step_end = time.perf_counter()
            mean_step_duration_across_gpus = step_end - step_start

            time_taken = step_end - step_start
            total_throughput = num_items_in_batch / time_taken

            loss_tensor = accum_loss

            accuracy = float("nan")
            if num_items_in_batch != 0:
                accuracy = step_correctly_predicted_tokens_count / num_items_in_batch

            current_lr = arguments.learning_rate
            if arguments.num_warmup_steps is not None:
                current_lr = scheduler.get_last_lr()[0]
            
            logging.info("epoch %d/%d, step %d/%d, mean step duration across gpus %.4f seconds, lr %.8f, loss %.4f, throughput %d tps, accuracy %.4f", epoch + 1, arguments.epochs, total_steps_passed, arguments.num_training_steps, mean_step_duration_across_gpus, current_lr, loss_tensor, total_throughput, accuracy)
            clearml_logger.report_scalar(title="train/steploss", series="series", value=loss_tensor, iteration=total_steps_passed)
            clearml_logger.report_scalar(title="train/throughput tokens/s", series="series", value=total_throughput, iteration=total_steps_passed)
            clearml_logger.report_scalar(title="train/stepaccuracy", series="series", value=accuracy, iteration=total_steps_passed)
            clearml_logger.report_scalar(title="train/epoch", series="series", value=epoch, iteration=total_steps_passed)
            clearml_logger.report_scalar(title="train/lr", series="series", value=current_lr, iteration=total_steps_passed)

            if total_steps_passed % arguments.save == 0:
                pathlib.Path(f"{arguments.cpdir}/epoch_{epoch}_step_{total_steps_passed}").mkdir(parents=True)
                safetensors.torch.save_file(model.state_dict(), f"{arguments.cpdir}/epoch_{epoch}_step_{total_steps_passed}/model.safetensors")
                with open(arguments.eagle_config_path, "r") as f:
                    config_data = json.load(f)
                config_data["architectures"] = ["LlamaForCausalLMEagle"]
                with open(f"{arguments.cpdir}/epoch_{epoch}_step_{total_steps_passed}/config.json", "w") as file:
                    json.dump(config_data, file, ensure_ascii=False, indent=4)        

            if total_steps_passed == arguments.num_training_steps:
                break

        if total_steps_passed == arguments.num_training_steps:
            break


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Coach that trains eagle draft model")
    parser.add_argument("--token-mapping-cache", type=pathlib.Path, required=True, help="Path to token mapping cache (d2t and t2d)")
    parser.add_argument("--micro-batch-size", type=int, required=True, help="Micro batch size")
    parser.add_argument("--gradient-accumulation-steps", type=int, required=True, help="Gradient accumulation steps")
    parser.add_argument("--num-warmup-steps", type=int, required=True, help="Num warmup steps")
    parser.add_argument("--num-training-steps", type=int, required=True, help="Num training steps")
    parser.add_argument("--clearml-project", type=str, required=True, help="Clearml project")
    parser.add_argument("--clearml-task", type=str, required=True, help="Clearml task")
    parser.add_argument("--epochs", type=int, required=True, help="Epochs")
    parser.add_argument("--verifier-model-path", type=str, required=True, help="verifier_model_path")
    parser.add_argument("--dataset-path", type=str, required=True, help="verifier_model_path")
    parser.add_argument("--eagle-config-path", type=str, required=True, help="eagle_config_path")
    parser.add_argument("--learning-rate", type=float, required=True, help="eagle_config_path")
    parser.add_argument("--maximum-model-length", type=int, required=True, help="eagle_config_path")
    parser.add_argument("--noise-low", type=float, required=True, help="eagle_config_path")
    parser.add_argument("--noise-high", type=float, required=True, help="eagle_config_path")
    parser.add_argument("--v-w", type=float, required=True, help="eagle_config_path")
    parser.add_argument("--p-w", type=float, required=True, help="eagle_config_path")
    parser.add_argument("--grad-clip", type=float, required=True, help="eagle_config_path")
    parser.add_argument("--b1", type=float, required=True, help="eagle_config_path")
    parser.add_argument("--b2", type=float, required=True, help="eagle_config_path")
    parser.add_argument("--cpdir", type=pathlib.Path, default="./checkpoints", help="Path to folder to save checkpoints")
    parser.add_argument("--save", type=int, required=False, help="Save model after every number of steps")
    parser.add_argument("--mixed-precision", type=str, required=False, help="Save model after every number of steps")
    parser.add_argument("--verifier-model-lm-head-dtype", type=str, required=False, help="Save model after every number of steps")
    parser.add_argument("--verifier-model-dtype", type=str, required=False, help="Save model after every number of steps")
    parser.add_argument("--eagle-dtype", type=str, required=False, help="Save model after every number of steps")
    parser.add_argument("--attn", type=str, required=False, help="Save model after every number of steps")
    return parser.parse_args()


def _count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _initialize_verifier_lm_head(verifier_path: pathlib.Path) -> torch.nn.Linear:
    with open(f"{verifier_path}/config.json", "r") as file:
        config = json.load(file)
    head = torch.nn.Linear(config["hidden_size"], config["vocab_size"], bias=False)
    with open(os.path.join(verifier_path, "model.safetensors.index.json"), "r") as f:
        index_json = json.loads(f.read())
    head_path = index_json["weight_map"]["lm_head.weight"]
    with safetensors.safe_open(os.path.join(verifier_path, head_path), framework="pt") as f:
        tensor = f.get_slice("lm_head.weight")[:, :config["hidden_size"]]
    head.weight.data = tensor
    head.eval()
    for param in head.parameters():
        param.requires_grad = False
    return head

class Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset: datasets.Dataset) -> None:
        self._dataset = dataset
    
    def __len__(self) -> int:
        return len(self._dataset)
    
    def __getitem__(self, index: int) -> dict:
        return self._dataset[index]


class Collator:
    def __init__(self, model_path) -> None:
        self._tokenizer = transformers.AutoTokenizer.from_pretrained(str(model_path), use_fast=True)
        self._tokenizer.pad_token = "[PAD]"
        self._tokenizer.pad_token_id = 0

    def __call__(self, features: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        result = self._tokenizer.apply_chat_template([m["messages"] for m in features], tokenize=True, add_generation_prompt=False, return_dict=True, return_assistant_tokens_mask=True, return_tensors="pt", padding=True)
        return {
            "input_ids": result["input_ids"],
            "loss_mask": result["assistant_masks"]
        }


def _make_eagle_input(batch, verifier_model, max_model_len, transform_uniform_low, transformer_uniform_high, device):
    input_ids = batch["input_ids"].to(device)[:, :max_model_len]
    loss_mask = batch["loss_mask"].to(device)[:, :max_model_len]

    with torch.no_grad():
        outs_big = verifier_model(input_ids, output_hidden_states=True, use_cache=False)
        hidden_state_big = outs_big.hidden_states[-1]
        hidden_state_big = _apply_noise_to_hidden_state(hidden_state_big, transform_uniform_low, transformer_uniform_high)
        T, L, D = hidden_state_big.shape
        target = hidden_state_big.new_zeros((T, L, D)) 
        target[:, :-1, :] = hidden_state_big[:, 1:, :]
        input_ids = torch.cat((input_ids[:, 1:], torch.zeros(input_ids.size(0), 1, dtype=input_ids.dtype, device=input_ids.device)), dim=1)
        batch = {"input_ids": input_ids, "hidden_states": hidden_state_big, "target": target, "loss_mask": loss_mask}
        return batch


def _apply_noise_to_hidden_state(hidden_state: torch.FloatTensor, transform_uniform_low, transformer_uniform_high) -> None:
    noise = torch.rand_like(hidden_state) * (transformer_uniform_high - transform_uniform_low) + transform_uniform_low
    noisy_tensor = hidden_state + noise
    return noisy_tensor


if __name__ == "__main__":
    coach()
