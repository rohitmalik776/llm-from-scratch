import os

import argparse
from datetime import datetime
from loguru import logger
from matplotlib import pyplot as plt
from matplotlib.pyplot import MaxNLocator
import tiktoken
import torch
from tqdm import tqdm
import yaml
import json

import gpt_model
import instruction_fine_tuning.instruction_dataset as instruction_dataset
import instruction_fine_tuning.evaluation as evaluation


logger.remove()
logger.add(tqdm.write)


figures_path = None
path_prefix = None


def load_model_tokenizer(config, model_size, device):
    tokenizer = tiktoken.get_encoding("gpt2")
    torch.manual_seed(123)
    model = gpt_model.GPTModel(config)
    gpt2_params = gpt_model.load_gpt2_params_from_tf_ckpt(
        f'./gpt2/{model_size}')
    gpt_model.load_weights_into_gpt(model, gpt2_params)
    del gpt2_params
    model.to(device)

    return model, tokenizer


def calc_loss_batch(model, inputs, labels, device):
    inputs = inputs.to(device)
    labels = labels.to(device)

    outputs = model(inputs)

    loss = torch.nn.functional.cross_entropy(
        outputs.flatten(0, 1), labels.flatten())

    return loss


def calc_loss_loader(model, dataloader, device):
    total_loss = 0.
    n_samples = len(dataloader)

    if n_samples <= 0:
        return -1

    for inputs, labels in tqdm(dataloader, desc='Calculating Loss', leave=False):
        total_loss += calc_loss_batch(model, inputs, labels, device).item()

    return total_loss / n_samples


def generate_preds_loader(model, dataloader):
    targets_lst = []
    outputs_lst = []

    for (inputs, targets) in tqdm(dataloader, desc='Evaluating', leave=False):
        outputs = model(inputs)

        outputs = torch.argmax(outputs, dim=-1)

        targets_lst.append(targets.cpu())
        outputs_lst.append(outputs.cpu())

    return targets_lst, outputs_lst


def evaluate_model_metrics(model, dataloader, tokenizer, return_txt=False):
    target_lst, output_lst = generate_preds_loader(model, dataloader)
    res = evaluation.evaluate_model(
        targets=target_lst,
        outputs=output_lst,
        tokenizer=tokenizer,
        return_txt=return_txt,
    )
    return res


def evaluate_model_loss(model, train_loader, val_loader, device):
    model.eval()
    train_loss = calc_loss_loader(model, train_loader, device)
    val_loss = calc_loss_loader(model, val_loader, device)
    model.train()

    return train_loss, val_loss


def train_model(model, train_loader, val_loader, optimizer, epochs, device, tokenizer, eval_step):
    hallucination_rates, failure_rates, json_formatting_rates, coverages = [], [], [], []
    train_losses, val_losses, steps = [], [], []
    global_step = -1

    total_steps = epochs * len(train_loader)
    pbar = tqdm(total=total_steps, desc='Training')

    model.train()
    for epoch in range(epochs):
        for x, y in train_loader:
            loss = calc_loss_batch(model, x, y, device)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.update(1)
            pbar.set_postfix({
                'epoch': epoch,
                'loss': loss.item(),
            })

            global_step += 1

            if global_step % eval_step == 0:
                model.eval()
                train_loss, val_loss = evaluate_model_loss(
                    model,
                    train_loader,
                    val_loader,
                    device,
                )

                metrics = evaluate_model_metrics(
                    model=model,
                    dataloader=val_loader,
                    tokenizer=tokenizer,
                )

                model.train()

                hr = metrics['hallucination_rate']
                fr = metrics['failure_rate']
                jr = metrics['json_formatting_rate']
                cr = metrics['coverage']

                logger.info(
                    f'Epoch: {epoch} - (Step: {global_step:06d})\n\t\tTrain loss: {train_loss}, Val loss: {val_loss}'
                    + f'\n\t\tHallucination rate: {hr}, Failure rate: {fr}\n\t\tJson Formatting rate: {jr}, Coverage: {cr}'
                )

                hallucination_rates.append(hr)
                failure_rates.append(fr)
                json_formatting_rates.append(jr)
                coverages.append(cr)

                train_losses.append(train_loss)
                val_losses.append(val_loss)
                steps.append(global_step)

    pbar.close()
    model.eval()

    metrics = {
        'hallucination_rate': hallucination_rates,
        'failure_rate': failure_rates,
        'json_formatting_rate': json_formatting_rates,
        'coverage': coverages,
    }

    return train_losses, val_losses, steps, metrics


def plot_losses(steps, tokens_seen, train_losses, val_losses):
    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax1.plot(steps, train_losses, label="Training loss")
    ax1.plot(
        steps, val_losses, linestyle="-.", label="Validation loss"
    )
    ax1.set_xlabel("Steps")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    # ax2 = ax1.twiny()
    # ax2.plot(tokens_seen, train_losses, alpha=0)
    # ax2.set_xlabel("Tokens seen")
    fig.tight_layout()
    plt.savefig(f'{figures_path}/training_losses.png')
    plt.close()


def plot_metrics(steps, metrics):
    fig, ax1 = plt.subplots(figsize=(8, 4))
    hallucination = metrics['hallucination_rate']
    failures = metrics['failure_rate']
    json_formatting = metrics['json_formatting_rate']
    coverage = metrics['coverage']

    ax1.plot(steps, hallucination, label='Hallucination')
    ax1.plot(steps, failures, label='Failure')
    ax1.plot(steps, json_formatting, label='JSON Formatting')
    ax1.plot(steps, coverage, label='Coverage')

    ax1.set_xlabel("Steps")
    ax1.set_ylabel("Rate")
    ax1.legend(loc="upper right")
    fig.tight_layout()
    plt.savefig(f'{figures_path}/training_metrics.png')
    plt.close()


def save_model_checkpoint(model):
    model_save_path = f"{path_prefix}/trained_model_checkpoint.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "config": config,
        },
        model_save_path,
    )
    logger.success(f"Saved trained model to: {model_save_path}")


def test_model(model, dataloader, tokenizer, device):
    test_loss = calc_loss_loader(model, dataloader, device)
    metrics, txt = evaluate_model_metrics(
        model, dataloader, tokenizer, return_txt=True)

    hr = metrics['hallucination_rate']
    fr = metrics['failure_rate']
    jr = metrics['json_formatting_rate']
    cr = metrics['coverage']

    logger.info(f'Test Results')
    logger.info(f'Loss: {test_loss}')
    logger.info(f'Hallucination Rate: {hr}')
    logger.info(f'Failure Rate: {fr}')
    logger.info(f'JSON Formattting Rate: {jr}')
    logger.info(f'Coverage: {cr}')

    with open(f"{path_prefix}/test_preds.json", "w") as outfile:
        json.dump(txt, outfile, indent=4)


def main(config: dict):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = tiktoken.get_encoding("gpt2")

    model_config = config['model']
    data_config = config['data']
    train_config = config['train']

    train_loader, test_loader, val_loader = instruction_dataset.get_instruction_dataloaders(
        tokenizer=tokenizer,
        batch_size=data_config['batch_size'],
        device=device,
        pin_memory=False,
        max_length=data_config['max_length'],
        split=data_config['split'],
        figures_path=figures_path,
    )

    model_size = model_config['param_count']

    model, tokenizer = load_model_tokenizer(model_config, model_size, device)
    logger.success(f'loaded model!')
    logger.info(f'model: \n{model}')

    model.eval()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config['learning_rate'],
        weight_decay=train_config['weight_decay'],
    )

    train_losses, val_losses, steps, metrics = train_model(
        model,
        train_loader,
        val_loader,
        optimizer,
        train_config['epochs'],
        device,
        tokenizer,
        train_config['eval_step'],
    )

    plot_losses(steps, None, train_losses, val_losses)
    plot_metrics(steps, metrics)

    save_model_checkpoint(model)
    test_model(model, test_loader, tokenizer, device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='Config file for model training')
    args = parser.parse_args()
    logger.info(f'Config file path:\n{args.config}')

    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError as e:
        logger.error(f"Config file not found! error: {e}")
        exit()
    except Exception as e:
        logger.error(f'Unknown exception occured: {e}')
        exit()

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    path_prefix = '.'.join(args.config.split('.')[:-1]).split('/')[-1]
    path_prefix = f'./output/{path_prefix}_{timestamp}'
    if os.path.exists(path_prefix):
        raise Exception('Run folder already exists!')
        exit()
    else:
        figures_path = f'{path_prefix}/figures'
        log_path = f'{path_prefix}/train.log'
        os.makedirs(figures_path)
        logger.add(log_path)

    logger.info(f'Config:\n{config}')

    main(config)
