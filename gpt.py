import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm.auto import tqdm
import argparse
import logging
import os

# ------------ Hyperparameters ------------
BATCH_SIZE: int = 64
BLOCK_SIZE: int = 296
MAX_ITERS: int = 1000
LEARNING_RATE: float = 5e-4
DEVICE: str = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
EVAL_INTERVAL_FACTOR: int = 10  # Higher is more here
EVAL_ITERS: int = 30
N_EMBEDDING: int = 384
N_HEAD: int = 6
N_LAYER: int = 6
DROPOUT: float = 0.2
MODEL_SAVE_PATH: str = 'gpt_model.pth'
# -----------------------------------------

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_text_data(file_path: str) -> str:
    """Load text data from a given file path."""
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    logging.info(f"Dataset loaded. Total characters: {len(text)}")
    return text

def prepare_vocab(text: str):
    """Prepare character-level vocabulary."""
    chars = sorted(set(text))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    logging.info(f"Vocabulary size: {vocab_size}")
    return stoi, itos, vocab_size

def encode(text: str, stoi: dict) -> list[int]:
    return [stoi[c] for c in text]

def decode(indices: list[int], itos: dict) -> str:
    return ''.join([itos[i] for i in indices])

def get_batch(split: str, data: torch.Tensor, train_data: torch.Tensor, val_data: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Retrieve a batch of training or validation data."""
    data_split = train_data if split == 'train' else val_data
    indices = torch.randint(len(data_split) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack([data_split[i:i+BLOCK_SIZE] for i in indices])
    y = torch.stack([data_split[i+1:i+BLOCK_SIZE+1] for i in indices])
    return x.to(DEVICE), y.to(DEVICE)

@torch.no_grad()
def estimate_loss(model: nn.Module, train_data: torch.Tensor, val_data: torch.Tensor) -> dict:
    """Estimate loss on training and validation data."""
    model.eval()
    losses = {'train': [], 'val': []}
    for split in ['train', 'val']:
        for _ in range(EVAL_ITERS):
            x, y = get_batch(split, None, train_data, val_data)
            _, loss = model(x, y)
            losses[split].append(loss.item())
    model.train()
    return {k: sum(v)/len(v) for k, v in losses.items()}

class GPTLanguageModel(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, N_EMBEDDING)
        self.position_embedding_table = nn.Embedding(BLOCK_SIZE, N_EMBEDDING)
        self.blocks = nn.Sequential(
            *[nn.TransformerEncoderLayer(d_model=N_EMBEDDING, nhead=N_HEAD, dropout=DROPOUT) for _ in range(N_LAYER)]
        )
        self.ln_f = nn.LayerNorm(N_EMBEDDING)
        self.lm_head = nn.Linear(N_EMBEDDING, vocab_size)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None):
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(idx.shape[1], device=DEVICE))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), targets.view(-1)) if targets is not None else None
        return logits, loss

    def generate(self, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        """Generate new text given an initial input."""
        with tqdm(range(max_new_tokens), dynamic_ncols=True, desc="Generating") as pbar:
            for _ in pbar:
                idx_cond = idx[:, -BLOCK_SIZE:]
                logits, _ = self(idx_cond)
                logits = logits[:, -1, :]
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
                idx = torch.cat((idx, idx_next), dim=1)
        return idx

def train():
    """Train the GPT model."""
    text = load_text_data('data/input.txt')
    stoi, itos, vocab_size = prepare_vocab(text)
    data = torch.tensor(encode(text, stoi), dtype=torch.long)
    train_data, val_data = data[:int(0.9 * len(data))], data[int(0.9 * len(data)):] 
    model = GPTLanguageModel(vocab_size).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    logging.info(f"Model Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.6f}M")
    logging.info(f"Training started for {MAX_ITERS} iterations...")

    for iteration in tqdm(range(MAX_ITERS), dynamic_ncols=True):
        xb, yb = get_batch('train', data, train_data, val_data)
        _, loss = model(xb, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if iteration % (MAX_ITERS // EVAL_INTERVAL_FACTOR) == 0:
            losses = estimate_loss(model, train_data, val_data)
            logging.info(f"Iteration {iteration}: Train Loss: {losses['train']:.4f}, Val Loss: {losses['val']:.4f}")

    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    logging.info(f"Model saved at {MODEL_SAVE_PATH}")

def generate():
    """Load trained model and generate text."""
    text = load_text_data('data/input.txt')
    _, itos, vocab_size = prepare_vocab(text)
    model = GPTLanguageModel(vocab_size).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    model.eval()
    context = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)
    generated_text = decode(model.generate(context, max_new_tokens=200)[0].tolist(), itos)
    print("\nGenerated Text:")
    print(generated_text)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "generate"], required=True)
    args = parser.parse_args()
    train() if args.mode == "train" else generate()