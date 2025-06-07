import os
import json
import torch
import typing
import pathlib
import clearml
import argparse
import accelerate
import safetensors
import transformers

from eagle.model import Model
from eagle.dataset import Dataset
from eagle.collator import Collator


def _train() -> None:
    arguments = _parse_arguments()
    
    train_dataset_path: pathlib.Path = arguments.train_input
    test_dataset_path: pathlib.Path = arguments.test_input
    model_path: pathlib.Path = arguments.model
    max_model_len = arguments.max_model_len
    epochs = arguments.epochs
    steps = arguments.steps
    warmup_steps: typing.Optional[int] = arguments.warmup_steps
    v_w = 1.0
    p_w = 0.1
    grad_clip = 0.5
    save_freq_steps = arguments.save
    cpdir = arguments.cpdir
    eagle_config_path = arguments.eagle_config
    lr = arguments.lr
    micro_bs = arguments.micro_bs
    clearml_project_name = arguments.project
    clearml_task_name = arguments.task
    evaluate_every_steps: int = arguments.evaluate
    state: pathlib.Path = arguments.state

    print("Initializing accelerate")
    torch.backends.cuda.matmul.allow_tf32 = True
    accelerate.utils.set_seed(0)
    accelerator = accelerate.Accelerator()
    
    if accelerator.is_main_process:
        print("Initializing clearml")
        clearml_task = clearml.Task.init(project_name=clearml_project_name, task_name=clearml_task_name)
        clearml_logger = clearml_task.get_logger()

    print("Initializing lm head")
    lm_head = _initialize_verifier_lm_head(verifier_path=model_path)
    # print(next(lm_head.parameters()).dtype)

    print("Initializing datasets")
    train_dataset = Dataset(dataset_path=train_dataset_path, max_model_len=max_model_len)
    test_dataset = Dataset(dataset_path=test_dataset_path, max_model_len=max_model_len)
    train_data_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=micro_bs, num_workers=0, collate_fn=Collator())
    test_data_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=micro_bs, num_workers=0, collate_fn=Collator())
    
    print("Initializing eagle model")
    config = transformers.AutoConfig.from_pretrained(eagle_config_path)
    model = Model(config, load_emb=True, path=model_path).to(config.torch_dtype)
    accelerator.register_for_checkpointing(model)
    # for k, v in model.named_parameters():
        # print(k, v.dtype)

    print("Initializing criterion, optimizer, lr scheduler")
    criterion = torch.nn.SmoothL1Loss(reduction="none")
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95))
    accelerator.register_for_checkpointing(optimizer)
    if warmup_steps is not None:
        scheduler = transformers.get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=steps
        )
        accelerator.register_for_checkpointing(scheduler)
        model, lm_head, optimizer, train_data_loader, test_data_loader, scheduler = accelerator.prepare(
            model, lm_head, optimizer, train_data_loader, test_data_loader, scheduler
        )
    else:
        model, lm_head, optimizer, train_data_loader, test_data_loader = accelerator.prepare(
            model, lm_head, optimizer, train_data_loader, test_data_loader
        )

    if state is not None:
        print(f"Loading state from {state}")
        accelerator.load_state(state)

    model.train()

    steps = 0

    print("Starting training loop")
    for epoch in range(epochs):
        model.train()
        num_batches = 0
        epoch_sum_loss = 0.0

        # Accuracy
        epoch_correctly_predicted_tokens_count = 0
        epoch_total_tokens_to_predict_count = 0

        for batch in train_data_loader:
            with accelerator.accumulate(model):
                optimizer.zero_grad()
                predict = model(batch["hidden_states"], input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])

                with torch.no_grad():
                    target_head = lm_head(batch["target"])
                    target_p = torch.nn.Softmax(dim=2)(target_head)
                    target_p = target_p.detach()  # bs, seq_len, vocab_size
                
                out_head = lm_head(predict)
                out_logp = torch.nn.LogSoftmax(dim=2)(out_head)  # bs, seq_len, vocab_size

                # Accuracy
                step_accuracy, correctly_predicted_tokens_count, total_tokens_to_predict = _compute_accuracy(
                    target_probabilities=target_p, 
                    predicted_probabilities=out_logp, 
                    loss_mask=batch["loss_mask"]
                )
                # If none means loss mask contained 0 only thus we can not compute accuracy and report it
                if step_accuracy is not None:
                    epoch_correctly_predicted_tokens_count += correctly_predicted_tokens_count
                    epoch_total_tokens_to_predict_count += total_tokens_to_predict

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
                steps += 1
                if warmup_steps is not None:
                    scheduler.step()
                    if accelerator.is_local_main_process:
                        print('[Train] Epoch {}/{}, Step {}, lr: {:.8f}'.format(epoch + 1, epochs, steps, scheduler.get_last_lr()[0]))
                        clearml_logger.report_scalar(title="lr", series="series", value=scheduler.get_last_lr()[0], iteration=steps)

                if step_accuracy is not None:
                    if accelerator.is_local_main_process:
                        print('[Train] Epoch {}/{}, Step {}, Step loss: {:.4f}, Step accuracy: {:.4f}'.format(epoch + 1, epochs, steps, loss.item(), step_accuracy))
                        clearml_logger.report_scalar(title="train/steploss", series="series", value=loss.item(), iteration=steps)
                        clearml_logger.report_scalar(title="train/stepaccuracy", series="series", value=step_accuracy, iteration=steps)
                else:
                    if accelerator.is_local_main_process:
                        print('[Train] Epoch {}/{}, Step {}, Step loss: {:.4f}'.format(epoch + 1, epochs, steps, loss.item()))
                        clearml_logger.report_scalar(title="train/steploss", series="series", value=loss.item(), iteration=steps)

                if accelerator.is_local_main_process and steps % save_freq_steps == 0:
                    _save_vllm_checkpoint(
                        accelerator=accelerator,
                        model=model,
                        eagle_config_path=eagle_config_path,
                        save_directory=f"{cpdir}/step_{steps}/vllm"
                    )

                    _save_checkpoint_state(
                        accelerator=accelerator,
                        model=model,
                        save_directory=f"{cpdir}/step_{steps}/state"
                    )
                
                if accelerator.is_local_main_process and steps % evaluate_every_steps == 0:
                    _eval(
                        model=model,
                        lm_head=lm_head,
                        test_data_loader=test_data_loader,
                        criterion=criterion,
                        clearml_logger=clearml_logger,
                        v_w=v_w,
                        p_w=p_w,
                        epoch=epoch,
                        epochs=epochs,
                        steps=steps,
                        accelerator=accelerator
                    )
        
        epoch_mean_loss = epoch_sum_loss / num_batches

        # Accuracy
        if epoch_total_tokens_to_predict_count == 0:
            epoch_mean_accuracy = None
        else:
            epoch_mean_accuracy = epoch_correctly_predicted_tokens_count / epoch_total_tokens_to_predict_count

        if accelerator.is_local_main_process:
            # Always report last training metrics
            clearml_logger.report_scalar(title="train/epochloss", series="series", value=epoch_mean_loss, iteration=epoch + 1)
            if epoch_mean_accuracy is not None:
                print('[Train] Epoch {}/{}, Epoch loss: {:.4f}, Epoch accuracy: {:.4f}'.format(epoch + 1, epochs, epoch_mean_loss, epoch_mean_accuracy))
                clearml_logger.report_scalar(title="train/epochaccuracy", series="series", value=epoch_mean_accuracy, iteration=epoch + 1)
            else:
                print('[Train] Epoch {}/{}, Epoch loss: {:.4f}'.format(epoch + 1, epochs, epoch_mean_loss))
            
    
    if accelerator.is_local_main_process:
        # Always eval in the end of the tarining
        _eval(
            model=model,
            lm_head=lm_head,
            test_data_loader=test_data_loader,
            criterion=criterion,
            clearml_logger=clearml_logger,
            v_w=v_w,
            p_w=p_w,
            epoch=epochs - 1,
            epochs=epochs,
            steps=steps,
            accelerator=accelerator
        )

        # Always save last checkpoint
        _save_vllm_checkpoint(
            accelerator=accelerator,
            model=model,
            eagle_config_path=eagle_config_path,
            save_directory=f"{cpdir}/step_{steps}/vllm"
        )

        _save_checkpoint_state(
            accelerator=accelerator,
            model=model,
            save_directory=f"{cpdir}/step_{steps}/state"
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
        "--evaluate",
        type=int,
        required=False,
        help="Evaluate model after every number of steps"
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


def _eval(
    model: torch.nn.Module, 
    lm_head: torch.nn.Module, 
    test_data_loader: torch.utils.data.DataLoader, 
    criterion: torch.nn.Module, 
    clearml_logger,
    v_w: float,
    p_w: float,
    epoch: int,
    epochs: int,
    steps: int,
    accelerator
) -> None:
    # Accuracy
    model.eval()
    eval_correctly_predicted_tokens_count = 0
    eval_total_tokens_to_predict_count = 0
    eval_loss_sum = 0.0
    eval_num_batches = 0
    for batch in test_data_loader:
        with torch.no_grad():
            predict = model(batch["hidden_states"], input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            target_head = lm_head(batch["target"])
            target_p = torch.nn.Softmax(dim=2)(target_head)
            target_p = target_p.detach()  # bs, seq_len, vocab_size
            out_head = lm_head(predict)
            out_logp = torch.nn.LogSoftmax(dim=2)(out_head)  # bs, seq_len, vocab_size
            # Accuracy
            _, correctly_predicted_tokens_count, total_tokens_to_predict = _compute_accuracy(
                target_probabilities=target_p, 
                predicted_probabilities=out_logp, 
                loss_mask=batch["loss_mask"]
            )
            if correctly_predicted_tokens_count is not None:
                eval_correctly_predicted_tokens_count += correctly_predicted_tokens_count
                eval_total_tokens_to_predict_count += total_tokens_to_predict

            loss_mask = batch["loss_mask"][:, :, None]
            plogp = target_p * out_logp
            ploss = -torch.sum(torch.sum(loss_mask * plogp, 2)) / (loss_mask.sum()+1e-5)
            vloss = criterion(predict, batch["target"])
            vloss = torch.sum(torch.mean(loss_mask * vloss, 2)) / (loss_mask.sum()+1e-5)
            loss = v_w * vloss + p_w * ploss
            eval_loss_sum += loss.item()
            eval_num_batches += 1

    val_mean_loss = eval_loss_sum / eval_num_batches
    if eval_total_tokens_to_predict_count != 0:
        val_mean_accuracy = eval_correctly_predicted_tokens_count / eval_total_tokens_to_predict_count
    else:
        val_mean_accuracy = None
    if accelerator.is_local_main_process:
        clearml_logger.report_scalar(title="validation/epochloss", series="series", value=val_mean_loss, iteration=epoch + 1)
        clearml_logger.report_scalar(title="validation/steploss", series="series", value=val_mean_loss, iteration=steps)
        if val_mean_accuracy is not None:
            print('[Validation] Epoch {}/{}, Step {}, Epoch loss: {:.4f}, Epoch accuracy: {:.4f}'.format(epoch + 1, epochs, steps, val_mean_loss, val_mean_accuracy))
            clearml_logger.report_scalar(title="validation/epochaccuracy", series="series", value=val_mean_accuracy, iteration=epoch + 1)
        else:
            print('[Validation] Epoch {}/{}, Step {}, Epoch loss: {:.4f}'.format(epoch + 1, epochs, steps, val_mean_loss))


if __name__ == "__main__":
    _train()
