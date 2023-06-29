from transformers import AdamW, get_linear_schedule_with_warmup
import torch
import numpy as np
from tqdm import tqdm
from config import Config


def setup_optimizer_scheduler(model, dataloader_train):
    """
    Setup the optimizer and scheduler for the training process.

    Args:
        model: The model for which the optimizer and scheduler will be set up.
        dataloader_train: The DataLoader for the training dataset.

    Returns:
        Tuple[AdamW, scheduler]: A tuple containing the optimizer and scheduler.
    """
    try:
        optimizer = AdamW(
            model.parameters(),
            lr=Config.lr,
            eps=Config.eps
        )

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=len(dataloader_train) * Config.epochs
        )

        return optimizer, scheduler

    except Exception as e:
        print(f"An error occurred while setting up the optimizer and scheduler: {e}")
        raise

import os
import torch
from tqdm import tqdm
from config import Config


def train(model, dataloader_train, optimizer, scheduler):
    """
    Approach adapted from an older version of HugginFace's ```run_glue.py``` script

    Train the model on the training dataset.

    Args:
        model: The model to train.
        dataloader_train: The DataLoader for the training dataset.
        optimizer: The optimizer for training.
        scheduler: The scheduler for training.
        device: The device to use for training.

    Returns:
        float: The average training loss.
    """
    device = Config.device
    model.to(device)
    print(f"using: {device}")
    model.train()
    loss_train_total = 0

    for epoch in range(1, Config.epochs + 1):
        if os.path.exists(f'Models/BERT_ft_last.model'):
            print(f"The Model Finished Training")
            return 0
        # Check if the model for this epoch already exists
        model_path = f'Models/BERT_ft_epoch{epoch}.model'
        if os.path.exists(model_path):
            print(f"Model for epoch {epoch} already exists. Skipping this epoch.")
            continue

        progress_bar = tqdm(
            dataloader_train,
            desc='Epoch {:d}'.format(epoch),
            leave=False,
            disable=False,
        )

        for batch in progress_bar:
            model.zero_grad()
            batch = tuple(b.to(device) for b in batch)
            inputs = {
                'input_ids': batch[0],
                'attention_mask': batch[1],
                'labels': batch[2],
            }
            outputs = model(**inputs)
            loss = outputs.loss
            loss_train_total += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            progress_bar.set_postfix({'training_loss': '{:.3f}'.format(loss.item() / len(batch))})
        if epoch ==  Config.epochs + 1:
            continue
        else:
            torch.save(model.state_dict(), model_path)
            tqdm.write('\nEpoch {}'.format(epoch))
            loss_train_avg = loss_train_total / len(dataloader_train)

    print('Finished Training The Model')
    # Save last model
    torch.save(model.state_dict(), f'Models/BERT_ft_last.model')
    return loss_train_avg


def evaluate(model, dataloader_val):
    """
    Evaluate the model on the validation dataset.

    Args:
        model: The model to evaluate.
        dataloader_val: The DataLoader for the validation dataset.

    Returns:
        Tuple[float, np.ndarray, np.ndarray]: A tuple containing the average validation loss, predicted labels, and true labels.
    """
    model.eval()
    device = Config.device
    loss_val_total = 0
    predictions, true_vals = [], []

    with torch.no_grad():
        for batch in tqdm(dataloader_val, desc="Evaluating"):
            batch = tuple(b.to(device) for b in batch)
            inputs = {
                'input_ids': batch[0],
                'attention_mask': batch[1],
                'labels': batch[2]
            }

            outputs = model(**inputs)
            loss = outputs[0]
            logits = outputs[1]
            loss_val_total += loss.item()

            logits = logits.detach().cpu().numpy()
            label_ids = inputs['labels'].cpu().numpy()
            predictions.append(logits)
            true_vals.append(label_ids)

    loss_val_avg = loss_val_total / len(dataloader_val)
    predictions = np.concatenate(predictions, axis=0)
    true_vals = np.concatenate(true_vals, axis=0)

    return loss_val_avg, predictions, true_vals

