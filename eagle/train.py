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
import accelerate
import contextlib
import safetensors
import transformers

# from eagle.model import Model
from eagle3.model import Model
from eagle3.configs import EConfig
import torch.nn.functional as F


def coach() -> None:
    arguments = _parse_arguments()

    accelerator = accelerate.Accelerator(log_with="all", gradient_accumulation_steps=arguments.gradient_accumulation_steps, mixed_precision=arguments.mixed_precision)
    accelerate.utils.set_seed(seed=0)
    torch.backends.cuda.matmul.allow_tf32 = True
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(asctime)s [%(filename)s:%(lineno)d] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    logger = accelerate.logging.get_logger(name=__name__, log_level="INFO")

    if accelerator.is_main_process:
        logger.info("Start to prepare clearml ", main_process_only=True)
        clearml_task = clearml.Task.init(project_name=arguments.clearml_project, task_name=arguments.clearml_task, reuse_last_task_id=False, continue_last_task=False, output_uri=False, auto_connect_frameworks=False, auto_resource_monitoring=False)
        clearml_logger = clearml_task.get_logger()
    
    logger.info("Start to prepare target model ", main_process_only=True)
    verifier_model = transformers.AutoModelForCausalLM.from_pretrained(arguments.verifier_model_path, device_map=accelerator.device, torch_dtype=getattr(torch, arguments.verifier_model_dtype), attn_implementation=arguments.attn)
    verifier_model = verifier_model.eval()
    verifier_model = verifier_model.to(accelerator.device)
    logger.info("Target model head has dtype %s", next(verifier_model.parameters()).dtype, main_process_only=True)
    logger.info("Target model head has %f billion parameters", _count_parameters(model=verifier_model) / 10 ** 9, main_process_only=True)
    if accelerator.is_main_process:
        clearml_logger.report_single_value(name="Target model head parameters billion", value=_count_parameters(model=verifier_model) / 10 ** 9)

    logger.info("Start to prepare draft model ", main_process_only=True)
    config = transformers.AutoConfig.from_pretrained(arguments.eagle_config_path)
    config.rope_scaling = {
        'type': 'linear',
        'factor': config.rope_scaling['factor']
    }
    eagle_config = EConfig(**config.to_dict())
    model = Model(eagle_config, load_emb=arguments.load_embeddings, path=arguments.verifier_model_path).to(getattr(torch, arguments.eagle_dtype)).to(accelerator.device)
    model.scandata(datapath=arguments.dataset_path, tokenizerpath=arguments.verifier_model_path, cachepath=arguments.cachepath)
    logger.info("Draft model head has dtype %s", next(model.parameters()).dtype, main_process_only=True)
    logger.info("Draft model head has %f billion parameters", _count_parameters(model=model) / 10 ** 9, main_process_only=True)
    model.train()
    accelerator.register_for_checkpointing(model)
    if accelerator.is_main_process:
        clearml_logger.report_single_value(name="Draft model head parameters billion", value=_count_parameters(model=model) / 10 ** 9)

    logger.info("Start to prepare data ", main_process_only=True)
    dataset = datasets.load_dataset("json", data_files={"train": [arguments.dataset_path]})["train"]
    dataset = Dataset(dataset=dataset)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=arguments.micro_batch_size, collate_fn=Collator(arguments.verifier_model_path))
    logger.info("Dataset contains %d samples", len(dataset), main_process_only=True)

    logger.info("Start to prepare miscellaneous ", main_process_only=True)
    model_optimizer = torch.optim.AdamW(model.parameters(), lr=arguments.learning_rate, betas=(arguments.b1, arguments.b2))
    accelerator.register_for_checkpointing(model_optimizer)

    scheduler = transformers.get_linear_schedule_with_warmup(optimizer=model_optimizer, num_warmup_steps=arguments.num_warmup_steps, num_training_steps=arguments.num_training_steps)
    accelerator.register_for_checkpointing(scheduler)

    model = accelerator.prepare_model(model)
    accelerator.skip_checkpointing(model.target_model)
    if arguments.load_embeddings:
        accelerator.skip_checkpointing(model.embed_tokens)
    model_optimizer = accelerator.prepare_optimizer(model_optimizer)
    dataloader = accelerator.prepare_data_loader(dataloader, device_placement=True)
    scheduler = accelerator.prepare_scheduler(scheduler)

    logger.info("Start training ", main_process_only=True)
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
            num_items_in_batch = accelerator.gather(num_items_in_batch).sum().item()

            for i, batch in enumerate(batch_samples):
                if (i < len(batch_samples) - 1 and accelerator.num_processes > 1):
                    ctx = model.no_sync
                else:
                    ctx = contextlib.nullcontext
                with ctx():
                    # batch = _make_eagle_input(batch, verifier_model, arguments.maximum_model_length, arguments.noise_low, arguments.noise_high, accelerator.device)
                    # batch["hidden_states"] = batch["hidden_states"]
                    # batch["target"] = batch["target"]
                    plosses, _, acces = model( 
                        input_ids=batch["input_ids"],
                        attention_mask=None,  # они сами потом её замутят
                        loss_mask=batch["loss_mask"]
                    )

                    ploss_weight = [0.8 ** i for i in range(len(plosses))]
                    ploss = sum([ploss_weight[i] * plosses[i] for i in range(len(plosses))])
                    loss = ploss

                    loss = (loss * arguments.gradient_accumulation_steps * accelerator.num_processes) / num_items_in_batch
                    accum_loss += loss.item()
                    accelerator.backward(loss)
                    accelerator.clip_grad_value_(model.parameters(), arguments.grad_clip)
            
            model_optimizer.step()
            scheduler.step()
            model_optimizer.zero_grad()
            
            total_steps_passed += 1

            step_end = time.perf_counter()
            step_duration = torch.tensor(step_end - step_start, device=accelerator.device)
            mean_step_duration_across_gpus = accelerator.reduce(step_duration, reduction="mean").item()

            time_taken = step_end - step_start
            throughput = torch.tensor(num_items_in_batch / time_taken, device=accelerator.device)
            total_throughput = accelerator.reduce(throughput, reduction="sum").item()

            loss_tensor = torch.tensor(accum_loss / (arguments.gradient_accumulation_steps * accelerator.num_processes), device=accelerator.device)
            loss_tensor = accelerator.reduce(loss_tensor, reduction="sum").item()

            current_lr = arguments.learning_rate
            if arguments.num_warmup_steps is not None:
                current_lr = scheduler.get_last_lr()[0]
            
            logger.info("epoch %d/%d, step %d/%d, mean step duration across gpus %.4f seconds, lr %.8f, loss %.4f, throughput %d tps, accuracy[-1] %.4f", epoch + 1, arguments.epochs, total_steps_passed, arguments.num_training_steps, mean_step_duration_across_gpus, current_lr, loss_tensor, total_throughput, acces[-1], main_process_only=True)
            if accelerator.is_main_process:
                clearml_logger.report_scalar(title="train/steploss", series="series", value=loss_tensor, iteration=total_steps_passed)
                clearml_logger.report_scalar(title="train/throughput tokens/s", series="series", value=total_throughput, iteration=total_steps_passed)
                clearml_logger.report_scalar(title="train/epoch", series="series", value=epoch, iteration=total_steps_passed)
                clearml_logger.report_scalar(title="train/lr", series="series", value=current_lr, iteration=total_steps_passed)
                for i, acc in enumerate(acces):
                    clearml_logger.report_scalar(title=f"train/stepaccuracy for step {i}", series="series", value=acc, iteration=total_steps_passed)

            if accelerator.is_local_main_process and total_steps_passed % arguments.save == 0:
                accelerator.save_state(output_dir=f"{arguments.cpdir}/epoch_{epoch}_step_{total_steps_passed}")
            

            if total_steps_passed == arguments.num_training_steps:
                break

        if total_steps_passed == arguments.num_training_steps:
            break

    accelerator.end_training()


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Coach that trains eagle draft model")
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
    parser.add_argument("--cachepath", type=pathlib.Path, required=False, help="cache of freq used tokens")
    parser.add_argument("--load-embeddings", type=bool, required=False, default=True, help="load embeddings or not")
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
        if self._tokenizer.pad_token is None or self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

    def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
        batch = {
            "id": [f["id"] for f in features],
            "conversations": [f["conversations"] for f in features],
        }

        preprocessed = self.preprocess_function(batch)
        # preprocessed = self.preprocess_function(features)

        input_ids = torch.cat(preprocessed["input_ids"], dim=0)
        loss_mask = torch.cat(preprocessed["loss_mask"], dim=0)

        return {
            "input_ids": input_ids,
            "loss_mask": loss_mask,
        }

    def preprocess_function(self, examples):
        tokenizer = self._tokenizer
        default_system_prompt = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."

        new_examples = {
            "input_ids": [],
            "loss_mask": []
        }

        for i in range(len(examples['id'])):
            messages = []
            convroles = ["user", "assistant"]
            roles = {"human": "user", "gpt": "assistant", "system": "system"}
            source = examples['conversations'][i]
            
            if not source:
                continue
            if roles[source[0]["from"]] != "user":
                if roles[source[0]["from"]] == "system":
                    system_promt = source[0]["value"]
                elif roles[source[0]["from"]] == "assistant":
                    system_promt = default_system_prompt
                
                messages.append({"role": "system", "content": system_promt})
                source = source[1:]

            for j, sentence in enumerate(source):
                role = roles[sentence["from"]]
                assert role == convroles[j % 2], f"{i}"
                messages.append({"role": role, "content": sentence["value"]})

            conversation = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )

            if not tokenizer.pad_token_id:
                tokenizer.pad_token_id = tokenizer.unk_token_id

            input_ids = tokenizer(
                conversation,
                return_tensors="pt",
                max_length=2048,
                add_special_tokens=False,
            ).input_ids[0]
            loss_mask = torch.ones_like(input_ids)

            sep = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            total_len = len(input_ids)
            sep2 = "<|eot_id|><|start_header_id|>user<|end_header_id|>"
            turns = conversation.split(sep2)

            turns[1] = turns[0] + sep2 + turns[1]
            turns = turns[1:]

            cur_len = 1
            loss_mask[:cur_len] = 0
            for i, turn in enumerate(turns):
                if turn == "":
                    break
                turn_len = len(tokenizer(turn).input_ids)

                parts = turn.split(sep)
                if len(parts) != 2:
                    break
                parts[0] += sep
                instruction_len = len(tokenizer(parts[0]).input_ids) - 1

                if i == 0:
                    loss_mask[cur_len: cur_len + instruction_len - 2] = 0
                else:
                    loss_mask[cur_len - 3: cur_len + instruction_len + 1] = 0
                cur_len += turn_len
                if i != 0:
                    cur_len += 3

            loss_mask[cur_len:] = 0

            new_examples["input_ids"].append(input_ids[None, :])
            new_examples["loss_mask"].append(loss_mask[None, :])
        
        max_len = max(x.shape[1] for x in new_examples["input_ids"])  # ?

        padded_input_ids = [
            F.pad(x, (0, max_len - x.shape[1]), value=self._tokenizer.pad_token_id)
            for x in new_examples["input_ids"]
        ]

        padded_loss_mask = [
            F.pad(x, (0, max_len - x.shape[1]), value=0)
            for x in new_examples["loss_mask"]
        ]

        return {
            "input_ids": padded_input_ids,
            "loss_mask": padded_loss_mask
        }


def _make_eagle_input(batch, verifier_model, max_model_len, transform_uniform_low, transformer_uniform_high, device):
    input_ids = batch["input_ids"].to(device, non_blocking=True)[:, :max_model_len]
    loss_mask = batch["loss_mask"].to(device, non_blocking=True)[:, :max_model_len]
    with torch.no_grad():
        outs_big = verifier_model(input_ids, output_hidden_states=True)
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
