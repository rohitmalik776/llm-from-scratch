import torch
from torch import nn
import tiktoken

from main import MultiHeadAttention


class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.tok_embedding = nn.Embedding(cfg['vocab_size'], cfg['emb_dim'])
        self.pos_embedding = nn.Embedding(cfg['context_length'], cfg['emb_dim'])
        self.emb_dropout = nn.Dropout(cfg['drop_rate'])

        self.trf_blocks = nn.Sequential(*[
            TransformerBlock(cfg) for _ in range(cfg['n_layers'])
        ])
        self.final_norm = LayerNorm(cfg['emb_dim'])
        self.out_head = nn.Linear(cfg['emb_dim'], cfg['vocab_size'], bias=False)


    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape

        tok_embeds = self.tok_embedding(in_idx)
        pos_embeds = self.pos_embedding(torch.arange(seq_len))

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
        self.self_attn = MultiHeadAttention(
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
        self.dropout2 = nn.Dropout(cfg['drop_rate'])

    def forward(self, x):
        shortcut = x
        
        x = self.norm1(x)
        x = self.self_attn(x)
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
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
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
    "qkv_bias": False
}

def main():
    tokenizer = tiktoken.get_encoding("gpt2")
    batch = []
    txt1 = "Arch is so cool"
    txt2 = "Especially with Gnome"
    batch.append(torch.tensor(tokenizer.encode(txt1)))
    batch.append(torch.tensor(tokenizer.encode(txt2)))

    batch = torch.stack(batch, dim=0)
    print(f'{batch = }')

    torch.manual_seed(123)
    model = GPTModel(GPT_CONFIG_124M)
    logits = model(batch)
    print("Output shape:", logits.shape)
    
    param_count = sum(p.numel() for p in model.parameters())
    print(f'Total parameter count of GPTModel: {param_count}')
    out_head_param_count = sum(p.numel() for p in model.out_head.parameters())
    print(f'out head param count: {out_head_param_count}')
    print(f'Total params without out head: {param_count - out_head_param_count}')

    total_bytes = param_count * 4
    total_size_mb = total_bytes / (1024 * 1024)
    print(f"Total size of the model: {total_size_mb:.2f} MB")




if __name__ == '__main__':
    main()