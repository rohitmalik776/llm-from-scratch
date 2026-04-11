import json
import os
import sys

import torch
from torch import nn
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import tiktoken
from tqdm import tqdm
import tensorflow as tf

from main import MultiHeadAttention
from gpt_model_data import get_verdict_dataloaders


LOAD_HF_GPT2 = False


class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.tok_embedding = nn.Embedding(cfg['vocab_size'], cfg['emb_dim'])
        self.pos_embedding = nn.Embedding(
            cfg['context_length'], cfg['emb_dim'])
        self.emb_dropout = nn.Dropout(cfg['drop_rate'])

        self.trf_blocks = nn.Sequential(*[
            TransformerBlock(cfg) for _ in range(cfg['n_layers'])
        ])
        self.final_norm = LayerNorm(cfg['emb_dim'])
        self.out_head = nn.Linear(
            cfg['emb_dim'], cfg['vocab_size'], bias=False)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape

        tok_embeds = self.tok_embedding(in_idx)
        pos_embeds = self.pos_embedding(
            torch.arange(seq_len, device=in_idx.device))

        x = tok_embeds + pos_embeds
        x = self.emb_dropout(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)

        return logits


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.norm1 = LayerNorm(cfg['emb_dim'])
        self.att = MultiHeadAttention(
            input_dim=cfg['emb_dim'],
            output_dim=cfg['emb_dim'],
            context_len=cfg['context_length'],
            dropout=cfg['drop_rate'],
            num_heads=cfg['n_heads'],
            qkv_bias=cfg['qkv_bias']
        )

        self.drop_shortcut = nn.Dropout(cfg['drop_rate'])

        self.norm2 = LayerNorm(cfg['emb_dim'])
        self.ffn = FeedForward(cfg)

    def forward(self, x):
        shortcut = x

        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        return x


class LayerNorm(nn.Module):
    def __init__(self, emb_dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)

        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            (2.0 / torch.pi) *
            (x + 0.044715 * torch.pow(x, 3))
        ))


def plot_relu_vs_gelu():
    import matplotlib.pyplot as plt

    relu = nn.ReLU()
    gelu = GELU()

    x = torch.linspace(-3, 3, 100)
    y_gelu, y_relu = gelu(x), relu(x)

    plt.figure(figsize=(8, 3))
    for i, (y, label) in enumerate(zip([y_gelu, y_relu], ["GELU", "RELU"]), 1):
        plt.subplot(1, 2, i)
        plt.plot(x, y)
        plt.title(f"{label} activation function")
        plt.xlabel("x")
        plt.ylabel(f"{label}(x)")
        plt.grid(True)
    plt.tight_layout()
    plt.savefig('./gelu_vs_relu.png')


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        in_dim = cfg['emb_dim']
        hid_dim = cfg['emb_dim'] * 4
        out_dim = cfg['emb_dim']

        self.layers = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            GELU(),
            nn.Linear(hid_dim, out_dim)
        )

    def forward(self, x):
        return self.layers(x)


GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": True
}

GPT_CONFIG_355M = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 1024,
    "n_heads": 16,
    "n_layers": 24,
    "drop_rate": 0.0,
    "qkv_bias": True,
}


def generate_text(model, idx, max_new_tokens, context_size, top_k=None, temperature=1.0):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)

        logits = logits[:, -1, :]

        if top_k is not None:
            top_k_vals, _ = torch.topk(logits, top_k)
            smallest_value = top_k_vals[:, -1]

            logits = torch.where(
                logits < smallest_value,
                -torch.inf,
                logits,
            )

        logits /= temperature

        probas = torch.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probas, 1)

        idx = torch.cat((idx, idx_next), dim=1)

    return idx


def text_to_token_ids(text, tokenizer):
    encodings = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encodings = torch.tensor(encodings).unsqueeze(0)
    return encodings


def token_ids_to_text(token_ids, tokenizer):
    token_ids = token_ids.squeeze(0)
    text = tokenizer.decode(token_ids.tolist())
    return text


def calc_loss_batch(inputs, targets, model, device):
    inputs = inputs.to(device)
    targets = targets.to(device)

    outputs = model(inputs)

    loss = torch.nn.functional.cross_entropy(
        input=outputs.flatten(0, 1),
        target=targets.flatten(),
    )

    return loss


def calc_loss_loader(dataloader, model, device):
    loss = 0.
    total_samples = len(dataloader)

    if total_samples <= 0:
        return -1

    for inputs, targets in dataloader:
        loss += calc_loss_batch(inputs, targets, model, device).item()

    return loss / total_samples


def generate_and_print_sample(model, start_context, tokenizer, device):
    model.eval()

    token_ids = text_to_token_ids(start_context, tokenizer).to(device)
    context_len = model.pos_embedding.weight.shape[0]
    with torch.no_grad():
        output = generate_text(model, token_ids, 100,
                               context_len, top_k=5, temperature=0.8)
    output_text = token_ids_to_text(output, tokenizer)

    print(f'Output text: {output_text.replace("\n", " ")}')
    model.train()


def evaluate_model(model, train_loader, val_loader, device):

    model.eval()

    train_loss = calc_loss_loader(train_loader, model, device)
    val_loss = calc_loss_loader(val_loader, model, device)

    model.train()

    return train_loss, val_loss


def train_model_simple(model, tokenizer, train_loader, val_loader, epochs, optimizer: torch.optim.Optimizer, sample_text, device):
    global_step = -1
    tokens_seen = 0

    train_losses = []
    val_losses = []
    track_tokens_seen = []

    total_steps = epochs * len(train_loader)

    pbar = tqdm(total=total_steps, desc='Training')

    for epoch in range(epochs):
        model.train()

        for (inputs, targets) in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(inputs, targets, model, device)
            loss.backward()
            optimizer.step()

            tokens_seen += inputs.numel()
            pbar.update(1)
            pbar.set_postfix({
                "epoch": epoch,
                "loss": loss.item(),
            })

            global_step += 1
            if global_step % 5 == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device)
                tqdm.write(
                    f'\tEpoch: {epoch} - (Step: {global_step: 06d})\n\t\tTraining loss: {train_loss}, Validation loss: {val_loss}')
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)

        sys.stdout.flush()

    generate_and_print_sample(model, sample_text, tokenizer, device)

    pbar.close()

    return train_losses, val_losses, track_tokens_seen


def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
    fig, ax1 = plt.subplots(figsize=(5, 3))
    ax1.plot(epochs_seen, train_losses, label="Training loss")
    ax1.plot(
        epochs_seen, val_losses, linestyle="-.", label="Validation loss"
    )
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax2 = ax1.twiny()
    ax2.plot(tokens_seen, train_losses, alpha=0)
    ax2.set_xlabel("Tokens seen")
    fig.tight_layout()
    plt.savefig('./model_training_stats.png')


def save_model(model: nn.Module, path):
    torch.save(model.state_dict(), path)


def load_model_(model: nn.Module, path, device):
    try:
        state_dict = torch.load(path, map_location=device)
        model.load_state_dict(state_dict=state_dict)
        print('Loaded model successfully!')
    except FileNotFoundError as e:
        print(f'Error while loading model: {e}')


def assign(left, right):
    if left.shape != right.shape:
        raise ValueError(
            f"Shape mismatch. Left: {left.shape}, "
            "Right: {right.shape}"
        )
    return torch.nn.Parameter(torch.tensor(right))


def load_gpt2_params_from_tf_ckpt(model_dir):
    ckpt_path = tf.train.latest_checkpoint(model_dir)
    settings = json.load(
        open(os.path.join(model_dir, "hparams.json"), "r", encoding="utf-8")
    )

    # Initialize parameters dictionary with empty blocks for each layer
    params = {"blocks": [{} for _ in range(settings["n_layer"])]}

    # Iterate over each variable in the checkpoint
    for name, _ in tf.train.list_variables(ckpt_path):
        # Load the variable and remove singleton dimensions
        variable_array = np.squeeze(tf.train.load_variable(ckpt_path, name))

        # Process the variable name to extract relevant parts
        variable_name_parts = name.split("/")[1:]  # Skip the 'model/' prefix

        # Identify the target dictionary for the variable
        target_dict = params
        if variable_name_parts[0].startswith("h"):
            layer_number = int(variable_name_parts[0][1:])
            target_dict = params["blocks"][layer_number]

        # Recursively access or create nested dictionaries
        for key in variable_name_parts[1:-1]:
            target_dict = target_dict.setdefault(key, {})

        # Assign the variable array to the last key
        last_key = variable_name_parts[-1]
        target_dict[last_key] = variable_array

    return params


def load_weights_into_gpt(gpt, params):
    gpt.pos_embedding.weight = assign(gpt.pos_embedding.weight, params['wpe'])
    gpt.tok_embedding.weight = assign(gpt.tok_embedding.weight, params['wte'])
    for b in range(len(params["blocks"])):
        q_w, k_w, v_w = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis=-1
        )
        gpt.trf_blocks[b].att.W_query.weight = assign(
            gpt.trf_blocks[b].att.W_query.weight, q_w.T
        )
        gpt.trf_blocks[b].att.W_key.weight = assign(
            gpt.trf_blocks[b].att.W_key.weight, k_w.T
        )
        gpt.trf_blocks[b].att.W_value.weight = assign(
            gpt.trf_blocks[b].att.W_value.weight, v_w.T
        )
        q_b, k_b, v_b = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis=-1)
        gpt.trf_blocks[b].att.W_query.bias = assign(
            gpt.trf_blocks[b].att.W_query.bias, q_b
        )
        gpt.trf_blocks[b].att.W_key.bias = assign(
            gpt.trf_blocks[b].att.W_key.bias, k_b
        )
        gpt.trf_blocks[b].att.W_value.bias = assign(
            gpt.trf_blocks[b].att.W_value.bias, v_b
        )
        gpt.trf_blocks[b].att.out_proj.weight = assign(
            gpt.trf_blocks[b].att.out_proj.weight,
            params["blocks"][b]["attn"]["c_proj"]["w"].T
        )
        gpt.trf_blocks[b].att.out_proj.bias = assign(
            gpt.trf_blocks[b].att.out_proj.bias,
            params["blocks"][b]["attn"]["c_proj"]["b"]
        )
        gpt.trf_blocks[b].ffn.layers[0].weight = assign(
            gpt.trf_blocks[b].ffn.layers[0].weight,
            params["blocks"][b]["mlp"]["c_fc"]["w"].T
        )
        gpt.trf_blocks[b].ffn.layers[0].bias = assign(
            gpt.trf_blocks[b].ffn.layers[0].bias,
            params["blocks"][b]["mlp"]["c_fc"]["b"])
        gpt.trf_blocks[b].ffn.layers[2].weight = assign(
            gpt.trf_blocks[b].ffn.layers[2].weight,
            params["blocks"][b]["mlp"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].ffn.layers[2].bias = assign(
            gpt.trf_blocks[b].ffn.layers[2].bias,
            params["blocks"][b]["mlp"]["c_proj"]["b"])
        gpt.trf_blocks[b].norm1.scale = assign(
            gpt.trf_blocks[b].norm1.scale,
            params["blocks"][b]["ln_1"]["g"])
        gpt.trf_blocks[b].norm1.shift = assign(
            gpt.trf_blocks[b].norm1.shift,
            params["blocks"][b]["ln_1"]["b"])
        gpt.trf_blocks[b].norm2.scale = assign(
            gpt.trf_blocks[b].norm2.scale,
            params["blocks"][b]["ln_2"]["g"])
        gpt.trf_blocks[b].norm2.shift = assign(
            gpt.trf_blocks[b].norm2.shift,
            params["blocks"][b]["ln_2"]["b"])

    gpt.final_norm.scale = assign(gpt.final_norm.scale, params["g"])
    gpt.final_norm.shift = assign(gpt.final_norm.shift, params["b"])
    gpt.out_head.weight = assign(gpt.out_head.weight, params["wte"])


def main():
    tokenizer = tiktoken.get_encoding("gpt2")
    torch.manual_seed(123)
    model = GPTModel(GPT_CONFIG_124M)

    param_count = sum(p.numel() for p in model.parameters())
    print(f'Total parameter count of GPTModel: {param_count}')
    out_head_param_count = sum(p.numel() for p in model.out_head.parameters())
    print(f'out head param count: {out_head_param_count}')
    print(
        f'Total params without out head: {param_count - out_head_param_count}')

    total_bytes = param_count * 4
    total_size_mb = total_bytes / (1024 * 1024)
    print(f"Total size of the model: {total_size_mb:.2f} MB")

    print('=' * 20, f'MODEL BUILDING END', '=' * 20)
    print("=" * 20, "START OF MODEL TRAINING", "=" * 20)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader = get_verdict_dataloaders(
        batch_size=2,
        tokenizer=tokenizer,
        train_ratio=0.9,
        max_length=GPT_CONFIG_124M['context_length'],
        stride=GPT_CONFIG_124M['context_length'],
    )

    print(f'Tokens in train_loader: {len(train_loader)}')
    print(f'Tokens in val_loader: {len(val_loader)}')

    model.to(device)

    if LOAD_HF_GPT2:
        print(f'Loading trained GPT2 model from HuggingFace')
        gpt2_params = load_gpt2_params_from_tf_ckpt('./gpt2/124M')
        load_weights_into_gpt(model, gpt2_params)
        del gpt2_params
        model.to(device)
    else:
        print(f'Loading self-trained model!')
        load_model_(model, './saved_model.pth', device=device)

    generate_and_print_sample(
        model, "We should all try to be", tokenizer, device
    )

    # with torch.no_grad():
    #     print(calc_loss_loader(train_loader, model, device))
    #     print(calc_loss_loader(val_loader, model, device))

    if not LOAD_HF_GPT2:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=0.0004,
            weight_decay=0.1
        )

        num_epochs = 10

        train_losses, val_losses, tokens_seen = train_model_simple(
            model,
            tokenizer,
            train_loader,
            val_loader,
            num_epochs,
            optimizer,
            "This is a great opportunity to",
            device
        )

        epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
        plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)

        save_model(model, './saved_model.pth')


if __name__ == '__main__':
    main()
