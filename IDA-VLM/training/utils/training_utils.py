# Training utilities

import os
import gc
import torch
import numpy as np
import random
import time
from datetime import timedelta
from trl import SFTConfig, SFTTrainer
from transformers import TrainerCallback


def setup_seeds(seed):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def create_training_args(config):
    """Create SFTConfig for training arguments."""
    return SFTConfig(
        output_dir=config.run_name,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=1,
        gradient_checkpointing=True,
        optim="adamw_torch",
        learning_rate=config.learning_rate,
        # lr_scheduler_type="constant",
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        logging_steps=200,
        disable_tqdm=True,  # suppress per-second tqdm updates in the log
        # log_level="error",
        eval_steps=2000,
        eval_strategy="steps",
        save_strategy="no",
        save_steps=100,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        load_best_model_at_end=False,
        bf16=True,
        tf32=True,
        max_grad_norm=3,
        warmup_ratio=0.03,
        push_to_hub=False,
        report_to="tensorboard" if not config.test else 'none',
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
    )


def save_training_scripts(run_name):
    """Save training artifacts including the model state."""
    os.system(f'cp train.py "{run_name}/train.py"')
    os.system(f'cp replace.py "{run_name}/replace.py"')
    os.system(f'cp -r utils "{run_name}"')


def clear_memory():
    """Clear GPU memory and perform garbage collection."""
    # Delete variables if they exist in the current global scope
    if "inputs" in globals():
        del globals()["inputs"]
    if "model" in globals():
        del globals()["model"]
    if "processor" in globals():
        del globals()["processor"]
    if "trainer" in globals():
        del globals()["trainer"]
    if "peft_model" in globals():
        del globals()["peft_model"]
    if "bnb_config" in globals():
        del globals()["bnb_config"]
    time.sleep(2)

    # Garbage collection and clearing CUDA memory
    gc.collect()
    time.sleep(2)
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    time.sleep(2)
    gc.collect()
    time.sleep(2)

    print(f"GPU allocated memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    print(f"GPU reserved memory: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")


def compute_metrics(eval_preds):
    """Compute token accuracy on the evaluation set."""
    predictions, labels = eval_preds
    # predictions are already argmaxed by preprocess_logits_for_metrics
    labels = torch.tensor(labels)
    predictions = torch.tensor(predictions)
    mask = labels != -100
    correct = (predictions[mask] == labels[mask]).sum().item()
    total = mask.sum().item()
    return {"eval_token_accuracy": correct / total if total > 0 else 0.0}


def preprocess_logits_for_metrics(logits, labels):
    """Argmax logits before storing to avoid OOM during evaluation."""
    if isinstance(logits, tuple):
        logits = logits[0]
    return logits.argmax(dim=-1)


class PeriodicStatusCallback(TrainerCallback):
    """Print a timestamped status line every `report_every` steps."""

    def __init__(self, report_every=500):
        self.report_every = report_every
        self.t_start = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.t_start = time.time()
        print(f"[status] Training started. total_steps={state.max_steps}")

    def on_step_end(self, args, state, control, **kwargs):
        step = state.global_step
        if step == 0 or step % self.report_every != 0:
            return
        elapsed = time.time() - self.t_start
        rate = step / elapsed if elapsed > 0 else 0.0
        eta = (state.max_steps - step) / rate if rate > 0 else 0.0
        last_loss = None
        if state.log_history:
            for entry in reversed(state.log_history):
                if 'loss' in entry:
                    last_loss = entry['loss']
                    break
        loss_str = f"loss={last_loss:.4f}" if last_loss is not None else "loss=n/a"
        print(
            f"[status] step {step}/{state.max_steps}  "
            f"{loss_str}  "
            f"elapsed={timedelta(seconds=int(elapsed))}  "
            f"eta={timedelta(seconds=int(eta))}"
        )


class PerEpochCheckpointCallback(TrainerCallback):
    """Save the trainable-parameter state dict at the end of every epoch.

    Files land at ``<run_name>/ckpts/connector_epoch_<N>.pt``. The dict values
    share storage with the live model parameters (in-place updated by the
    optimizer), so each ``torch.save`` snapshots the values *at that epoch*.
    """

    def __init__(self, run_name, trainable_state_dict):
        self.ckpt_dir = os.path.join(run_name, 'ckpts')
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.trainable_state_dict = trainable_state_dict

    def on_epoch_end(self, args, state, control, **kwargs):
        # state.epoch at end of epoch N is N.0; round to handle float jitter.
        epoch = int(round(state.epoch))
        path = os.path.join(self.ckpt_dir, f'connector_epoch_{epoch}.pt')
        torch.save(self.trainable_state_dict, path)
        print(f"[status] Saved epoch-{epoch} checkpoint → {path}")


class BestCheckpointCallback(TrainerCallback):
    """Save the trainable-parameter state dict whenever eval_loss improves.

    Writes to ``<run_name>/ckpts/connector_best_step_<N>.pt`` after each
    evaluation that produces a new lowest eval_loss. The previous best file
    is removed so only the current best remains on disk.
    """

    def __init__(self, run_name, trainable_state_dict):
        self.ckpt_dir = os.path.join(run_name, 'ckpts')
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.trainable_state_dict = trainable_state_dict
        self.best_eval_loss = float('inf')
        self.best_path = None

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if not metrics or 'eval_loss' not in metrics:
            return
        eval_loss = metrics['eval_loss']
        if eval_loss < self.best_eval_loss:
            prev = self.best_eval_loss
            self.best_eval_loss = eval_loss
            step = state.global_step
            path = os.path.join(self.ckpt_dir, f'connector_best_step_{step}.pt')
            torch.save(self.trainable_state_dict, path)
            if self.best_path is not None and self.best_path != path and os.path.exists(self.best_path):
                os.remove(self.best_path)
            self.best_path = path
            prev_str = f"{prev:.4f}" if prev != float('inf') else "n/a"
            print(f"[status] New best eval_loss={eval_loss:.4f} (prev={prev_str}) "
                  f"at step {step} → {path}")


def run_training(config, train_dataset, eval_dataset, model, processor):
    """Main training function that orchestrates the entire training process."""
    from .collate_utils import collate_fn
    from .model_utils import setup_trainable_parameters

    # Setup
    setup_seeds(config.seed)

    # Create training arguments
    training_args = create_training_args(config)

    # Configure trainable parameters BEFORE constructing the trainer so we can
    # hand the resulting state-dict to the per-epoch checkpoint callback.
    trainable_state_dict = setup_trainable_parameters(model, config.training_parameters)

    # Setup trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=lambda examples: collate_fn(examples, config, processor, model),
        processing_class=processor.tokenizer,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        callbacks=[
            PeriodicStatusCallback(report_every=500),
            PerEpochCheckpointCallback(config.run_name, trainable_state_dict),
            BestCheckpointCallback(config.run_name, trainable_state_dict),
        ],
    )

    # backup training scripts
    # save_training_scripts(config.run_name)

    # Run training — track wall-clock time end-to-end
    t0 = time.time()
    trainer.train()
    total = time.time() - t0
    print(f"[status] Training finished. total_time={timedelta(seconds=int(total))} "
          f"({total:.1f} s)")

    # Stable canonical alias at the run root, kept for back-compat with older
    # test scripts that point at <run_dir>/connector.pt directly. We make it
    # a relative symlink to the latest connector_epoch_<N>.pt rather than a
    # second on-disk copy, so the file is byte-identical to the last epoch
    # without duplicating it. Symlink target is *relative* so the run dir
    # remains portable (rsync / move / archive don't break the link).
    import re as _re
    import glob as _glob
    ckpt_dir = os.path.join(config.run_name, 'ckpts')
    def _epoch_num(p):
        m = _re.match(r'connector_epoch_(\d+)\.pt', os.path.basename(p))
        return int(m.group(1)) if m else -1
    epoch_files = _glob.glob(os.path.join(ckpt_dir, 'connector_epoch_*.pt'))
    latest_epoch_ckpt = max(epoch_files, key=_epoch_num) if epoch_files else None

    connector_path = os.path.join(config.run_name, 'connector.pt')
    if os.path.lexists(connector_path):
        os.unlink(connector_path)
    if latest_epoch_ckpt is not None:
        rel_target = os.path.join('ckpts', os.path.basename(latest_epoch_ckpt))
        os.symlink(rel_target, connector_path)
        print(f"[status] connector.pt → {rel_target} "
              f"(symlink alias for back-compat; saves duplicating the file)")
    else:
        # No per-epoch checkpoint found (zero-epoch dry run, or callback
        # disabled). Fall back to a real write so consumers get something.
        torch.save(trainable_state_dict, connector_path)
        print(f"[status] No epoch checkpoint to alias; wrote {connector_path} directly")