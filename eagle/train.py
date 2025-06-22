import os
import time
import json
import torch
import typing
import pathlib
import logging
import clearml
import argparse
import datasets
import accelerate
import safetensors
import transformers

from eagle.model import Model


logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s %(asctime)s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


def _train() -> None:
    arguments = _parse_arguments()
    model_path: pathlib.Path = arguments.model
    dataset_path = arguments.dataset_path
    max_model_len = arguments.max_model_len
    epochs = arguments.epochs
    steps = arguments.steps
    warmup_steps: typing.Optional[int] = arguments.warmup_steps
    v_w = arguments.v_w
    p_w = arguments.p_w
    grad_clip = arguments.grad_clip    
    save_freq_steps = arguments.save
    cpdir = arguments.cpdir
    eagle_config_path = arguments.eagle_config
    lr = arguments.lr
    gradient_accumulation_steps = arguments.gradient_accumulation_steps
    
    accelerator = accelerate.Accelerator(gradient_accumulation_steps=gradient_accumulation_steps, mixed_precision=arguments.mixed_precision)
    accelerate.utils.set_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    
    if accelerator.is_main_process:
        logging.info("Started to initialize clearml")
        clearml_task = clearml.Task.init(project_name=arguments.project, task_name=arguments.task, reuse_last_task_id=False, continue_last_task=False, output_uri=False, auto_connect_frameworks=False, auto_resource_monitoring=False)
        clearml_logger = clearml_task.get_logger()

    if accelerator.is_local_main_process:
        logging.info("Started to load target langauge model head")
    lm_head = _initialize_verifier_lm_head(verifier_path=model_path).to(getattr(torch, arguments.model_dtype))
    if accelerator.is_local_main_process:
        logging.info("Target langauge model head has dtype %s", next(lm_head.parameters()).dtype)

    if accelerator.is_local_main_process:
        logging.info("Started to load langauge model")
    verifier_model = transformers.AutoModelForCausalLM.from_pretrained(model_path, device_map=accelerator.device, attn_implementation="flash_attention_2", torch_dtype=arguments.model_dtype)
    verifier_model = verifier_model.eval()
    if accelerator.is_local_main_process:
        logging.info("Target langauge model has dtype %s", next(verifier_model.parameters()).dtype)

    if accelerator.is_local_main_process:
        logging.info("Started to load train dataset")
    train_dataset = datasets.load_dataset("json", data_files={"train": [str(dataset_path)]})["train"].shuffle(seed=0)
    train_dataset = Dataset(dataset=train_dataset)
    if accelerator.is_local_main_process:
        logging.info("Train dataset has %d samples", len(train_dataset))

    if accelerator.is_local_main_process:
        logging.info("Started to create dataloaders")
    train_data_loader = torch.utils.data.DataLoader(prefetch_factor=4, pin_memory=True, dataset=train_dataset, batch_size=arguments.micro_bs, num_workers=16, collate_fn=BaseCollator(model_path))

    if accelerator.is_local_main_process:
        logging.info("Started to load eagle model")
    config = transformers.AutoConfig.from_pretrained(eagle_config_path)
    model = Model(config, load_emb=True, path=model_path).to(getattr(torch, arguments.eagle_dtype))
    accelerator.register_for_checkpointing(model)
    if accelerator.is_local_main_process:
        eagle_parameters_count_b = _count_parameters(model) / 10 ** 9
        logging.info("Eagle model has %f billion parameters", eagle_parameters_count_b)
        clearml_logger.report_single_value(
            name="Eagle parameters count in billions", 
            value=eagle_parameters_count_b
        )
    
    if accelerator.is_local_main_process:
        logging.info("Started to load criterion")
    criterion = torch.nn.SmoothL1Loss(reduction="none")

    if accelerator.is_local_main_process:
        logging.info("Started to load optimizer")
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(arguments.optimizer_beta_1, arguments.optimizer_beta_2))
    
    scheduler = None
    if warmup_steps is not None:
        if accelerator.is_local_main_process:
            logging.info("Started to load learning rate scheduler")
        scheduler = transformers.get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=steps
        )

    if accelerator.is_local_main_process:
        logging.info("Accelerator started to register optimizer for checkpointing")
    accelerator.register_for_checkpointing(optimizer)

    if warmup_steps is not None:
        if accelerator.is_local_main_process:
            logging.info("Accelerator started to register learning rate scheduler for checkpointing")
        accelerator.register_for_checkpointing(scheduler)

        if accelerator.is_local_main_process:
            logging.info("Accelerator started preparation")
        model, lm_head, optimizer, train_data_loader, scheduler = accelerator.prepare(
            model, lm_head, optimizer, train_data_loader, scheduler
        )
    else:
        if accelerator.is_local_main_process:
            logging.info("Accelerator started preparation")
        model, lm_head, optimizer, train_data_loader = accelerator.prepare(
            model, lm_head, optimizer, train_data_loader
        )

    if arguments.state is not None:
        if accelerator.is_local_main_process:
            logging.info(f"Started to load state from {arguments.state}")
        accelerator.load_state(arguments.state)

    model.train()
    total_steps_passed = 0
    if accelerator.is_local_main_process:
        logging.info("Started training")

    training_start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()

        for batch_index, raw in enumerate(train_data_loader, start=1):
            step_start = time.perf_counter()

            with accelerator.accumulate(model):
                batch = make_eagle_input(raw, verifier_model, max_model_len, arguments.noise_low, arguments.noise_high, accelerator.device)
                predict = model(batch["hidden_states"], input_ids=batch["input_ids"])
                with torch.no_grad():
                    target_p = lm_head(batch["target"]).softmax(dim=2).detach()
                out_head = lm_head(predict)
                out_logp = torch.nn.LogSoftmax(dim=2)(out_head)

                _, correctly_predicted_tokens_count, total_tokens_to_predict = _compute_accuracy(
                    target_probabilities=target_p, 
                    predicted_probabilities=out_logp, 
                    loss_mask=batch["loss_mask"]
                )

                loss_mask = batch["loss_mask"][:, :, None]
                plogp = target_p * out_logp
                ploss = -torch.sum(torch.sum(loss_mask * plogp, 2)) / (loss_mask.sum()+1e-5)
                vloss = criterion(predict, batch["target"])
                vloss = torch.sum(torch.mean(loss_mask * vloss, 2)) / (loss_mask.sum()+1e-5)
                loss = v_w * vloss + p_w * ploss
                accelerator.backward(loss)
                accelerator.clip_grad_value_(model.parameters(), grad_clip)
                optimizer.step()
                optimizer.zero_grad()
                total_steps_passed += 1
                if warmup_steps is not None:
                    scheduler.step()
            
            # time
            step_end = time.perf_counter()
            step_duration = torch.tensor(step_end - step_start, device=accelerator.device)
            mean_step_duration_across_gpus = accelerator.reduce(step_duration, reduction="mean").item()

            # loss
            loss_tensor = torch.tensor(loss.item(), device=accelerator.device)
            mean_loss = accelerator.reduce(loss_tensor, reduction="mean").item()

            # throughput
            processed_tokens = loss_mask.sum()
            time_taken = step_end - step_start
            throughput = processed_tokens / time_taken
            total_throughput = accelerator.reduce(throughput, reduction="sum").item()

            # accuracy
            total_number_of_not_nans = accelerator.reduce(torch.tensor(0 if correctly_predicted_tokens_count is None else 1, device=accelerator.device), reduction="sum").item()
            if total_number_of_not_nans == 0:
                accuracy == float("nan")
            else:
                correctly_predicted_tokens_count = accelerator.reduce(torch.tensor(correctly_predicted_tokens_count or 0, device=accelerator.device), reduction="sum").item()
                total_tokens_to_predict = accelerator.reduce(torch.tensor(total_tokens_to_predict or 0, device=accelerator.device), reduction="sum").item()
                accuracy = correctly_predicted_tokens_count / total_tokens_to_predict

            # log all
            if accelerator.is_local_main_process:
                logging.info("epoch %d/%d, step %d/%d, dataloader %d/%d, Mean step duration across gpus %.4f seconds, lr %.8f, loss %.4f, throughput %d tps, accuracy %.4f", epoch, epochs, total_steps_passed, steps, batch_index, len(train_data_loader), mean_step_duration_across_gpus, scheduler.get_last_lr()[0], mean_loss, total_throughput, accuracy)
                clearml_logger.report_scalar(title="lr", series="series", value=scheduler.get_last_lr()[0], iteration=total_steps_passed)
                clearml_logger.report_scalar(title="train/throughput tokens/s", series="series", value=total_throughput, iteration=total_steps_passed)
                clearml_logger.report_scalar(title="train/steploss", series="series", value=mean_loss, iteration=total_steps_passed)
                clearml_logger.report_scalar(title="train/stepaccuracy", series="series", value=accuracy, iteration=total_steps_passed)
                clearml_logger.report_scalar(title="train/epoch", series="series", value=epoch, iteration=total_steps_passed)

            if accelerator.is_local_main_process and total_steps_passed % save_freq_steps == 0:
                _save_checkpoint_state(
                    accelerator=accelerator,
                    model=model,
                    save_directory=f"{cpdir}/epoch_{epoch}_step_{total_steps_passed}"
                )

            if total_steps_passed == steps:
                break

        if total_steps_passed == steps:
            break
        
        # time
        epoch_end = time.perf_counter()
        epoch_duration = torch.tensor(epoch_end - epoch_start, device=accelerator.device)
        mean_epoch_duration_across_gpus = accelerator.reduce(epoch_duration, reduction="mean").item()
        if accelerator.is_local_main_process:
            logging.info("Mean epoch %d duration across GPUs %.4f seconds", epoch, mean_epoch_duration_across_gpus)

    # time
    training_end = time.perf_counter()
    training_duration = torch.tensor(training_end - training_start, device=accelerator.device)
    mean_training_duration_across_gpus = accelerator.reduce(training_duration, reduction="mean").item()
    if accelerator.is_local_main_process:
        logging.info("Mean traning duration across GPUs %.4f seconds", mean_training_duration_across_gpus)
    
    accelerator.end_training()


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Process a raw chat dataset using a verifier model and tokenizer."
    )
    parser.add_argument(
        "--model",
        type=pathlib.Path,
        required=True,
        help="Path to verifier model"
    )
    parser.add_argument(
        "--model-dtype",
        type=str,
        required=True,
        help="Path to verifier model"
    )
    parser.add_argument(
        "--eagle-dtype",
        type=str,
        required=True,
        help="Path to verifier model"
    )
    parser.add_argument(
        "--dataset-path",
        type=pathlib.Path,
        required=True,
        help="Path to dataset"
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
        default=1_000,
        help="Number of epochs to train"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=1_000,
        help="Number of steps to train"
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        required=False,
        default=1_000,
        help="Number of warmup steps to train"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--cpdir",
        type=pathlib.Path,
        default="./checkpoints",
        help="Path to folder to save checkpoints"
    )
    parser.add_argument(
        "--state",
        type=pathlib.Path,
        required=False,
        help="Path to folder to load checkpoint from if train from checkpoint"
    )
    parser.add_argument(
        "--save",
        type=int,
        required=False,
        help="Save model after every number of steps"
    )
    parser.add_argument(
        "--eagle-config",
        type=pathlib.Path,
        help="path to eagle config"
    )
    parser.add_argument(
        "--micro-bs",
        type=int,
        default=1,
        help="Micro batch size"
    )
    parser.add_argument(
        "--project",
        type=str,
        default="eagle",
        help="Clearml project name"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="task",
        help="Clearml task name"
    )
    parser.add_argument(
        "--noise-low",
        type=float,
        default=0.0,
        help="Uniform hiddenstate noise low"
    )
    parser.add_argument(
        "--noise-high",
        type=float,
        default=0.0,
        help="Uniform hiddenstate noise high"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="gradient_accumulation_steps"
    )
    parser.add_argument(
        "--v-w",
        type=float,
        default=1.0,
        help="gradient_accumulation_steps"
    )
    parser.add_argument(
        "--p-w",
        type=float,
        default=0.1,
        help="gradient_accumulation_steps"
    )
    parser.add_argument(
        "--grad-clip",
        type=float,
        default=0.5,
        help="gradient_accumulation_steps"
    )
    parser.add_argument(
        "--optimizer-beta-1",
        type=float,
        default=0.9,
        help="gradient_accumulation_steps"
    )
    parser.add_argument(
        "--optimizer-beta-2",
        type=float,
        default=0.95,
        help="gradient_accumulation_steps"
    )
    parser.add_argument(
        "--mixed-precision",
        type=str,
        help="mixed_precision"
    )
    return parser.parse_args()


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


def _save_checkpoint_state(
    accelerator: accelerate.Accelerator,
    model: torch.nn.Module,
    save_directory: pathlib.Path
) -> None:
    accelerator.save_state(output_dir=save_directory)


def _compute_accuracy(
    target_probabilities: torch.FloatTensor,  # bs, seq_len, vocab_size
    predicted_probabilities: torch.FloatTensor,  # bs, seq_len, vocab_size
    loss_mask: torch.LongTensor # bs, seq_len
) -> typing.Tuple[typing.Optional[float], typing.Optional[int], typing.Optional[int]]:
    total_tokens_to_predict = loss_mask.sum().item()
    # It can happen that when filterd by maximum model length we completely remove
    # assistant replies with loss and have only user replies for which we do not want
    # to compute loss. In order not to divide later in computation of accuracy by zero
    # we return None
    if total_tokens_to_predict == 0:
        return None, None, None
    _, target_max_p_tokens = torch.max(target_probabilities, 2)
    _, ealge_max_p_tokens = torch.max(predicted_probabilities, 2)
    correctly_predicted_tokens_count = ((target_max_p_tokens == ealge_max_p_tokens) * loss_mask.squeeze()).sum().item()
    step_accuracy = correctly_predicted_tokens_count / total_tokens_to_predict
    return step_accuracy, correctly_predicted_tokens_count, total_tokens_to_predict


def _count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def make_eagle_input(batch, verifier_model, max_model_len, transform_uniform_low, transformer_uniform_high, device):
    input_ids = batch["input_ids"].to(device, non_blocking=True)[:, :max_model_len]
    loss_mask = batch["loss_mask"].to(device, non_blocking=True)[:, :max_model_len]

    with torch.no_grad():
        outs_big = verifier_model(input_ids, output_hidden_states=True)
        hidden_state_big = outs_big.hidden_states[-1]
        noise = torch.empty_like(hidden_state_big).uniform_(transform_uniform_low, transformer_uniform_high)
        hidden_state_big.add_(noise)
        T, L, D = hidden_state_big.shape
        target = hidden_state_big.new_zeros((T, L, D)) 
        target[:, :-1, :] = hidden_state_big[:, 1:, :]
        batch = {
            "input_ids": input_ids,
            "hidden_states": hidden_state_big,
            "target": target,
            "loss_mask": loss_mask
        }
        return batch


class Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset: datasets.Dataset) -> None:
        self._dataset = dataset
    
    def __len__(self) -> int:
        return len(self._dataset)
    
    def __getitem__(self, index: int) -> dict:
        return self._dataset[index]


class BaseCollator:
    def __init__(self, model_path) -> None:
        self._tokenizer = transformers.AutoTokenizer.from_pretrained(str(model_path), use_fast=True)
        self._tokenizer.pad_token = "[PAD]"

    def __call__(self, features: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        result = self._tokenizer.apply_chat_template(
            [m["messages"] for m in features], 
            tokenize=True, 
            add_generation_prompt=False, 
            return_dict=True,
            return_assistant_tokens_mask=True,
            return_tensors="pt",
            padding=True
        )
        return {
            "input_ids": result["input_ids"],
            "loss_mask": result["assistant_masks"]
        }


if __name__ == "__main__":
    _train()
