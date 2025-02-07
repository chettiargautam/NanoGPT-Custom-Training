import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import time
from tqdm import tqdm

# Hyperparameters
torch.manual_seed(1337)
BATCH_SIZE: int = 32  # Number of sequences to process in parallel
BLOCK_SIZE: int = 32    # Maximum context length for predictions
MAX_ITERATIONS: int = 25000
EVALUATION_INTERVAL: int = 500
LEARNING_RATE: float = 1e-3
EVALUATION_ITERATIONS: int = 200
EMBEDDING_DIM: int = 64  # Dimension for embeddings

def get_device() -> str:
    """Returns the best available compute device."""
    if torch.cuda.is_available():
        print(">>> Using CUDA")
        return 'cuda'
    elif torch.backends.mps.is_available():
        print(">>> Using MPS")
        return 'mps'
    else:
        print(">>> Using CPU")
        return 'cpu'

device = get_device()

def load_text_data(filepath: str) -> str:
    """Loads text data from a file."""
    print(f">>> Loading text data from {filepath}")
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()

def create_vocab_mapping(text: str):
    """Creates mappings between characters and their respective indices."""
    characters = sorted(list(set(text)))
    vocab_size = len(characters)
    stoi = {char: idx for idx, char in enumerate(characters)}
    itos = {idx: char for idx, char in enumerate(characters)}
    return vocab_size, stoi, itos

def encode(text: str, stoi: dict) -> list:
    return [stoi[char] for char in text]

def decode(indices: list, itos: dict) -> str:
    return ''.join([itos[i] for i in indices])

def prepare_data(encoded_text: torch.Tensor, train_ratio: float = 0.85):
    """Splits the data into training and test sets."""
    split_index = int(train_ratio * len(encoded_text))
    return encoded_text[:split_index], encoded_text[split_index:]

def get_batch(data: torch.Tensor) -> tuple:
    """Generates a batch of inputs and targets."""
    indices = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))
    x_batch = torch.stack([data[i:i + BLOCK_SIZE] for i in indices])
    y_batch = torch.stack([data[i + 1:i + BLOCK_SIZE + 1] for i in indices])
    return x_batch.to(device), y_batch.to(device)

class TrigramLanguageModel(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, EMBEDDING_DIM)
        self.trigram = nn.Linear(EMBEDDING_DIM * 3, vocab_size)  # Trigram linear layer

    def forward(self, indices: torch.Tensor, targets: torch.Tensor = None):
        embeddings = self.token_embedding(indices)  # (B, T, C)
        trigram_input = torch.cat([embeddings[:, :-2, :], embeddings[:, 1:-1, :], embeddings[:, 2:, :]], dim=-1)
        logits = self.trigram(trigram_input)

        loss = None
        if targets is not None:
            logits = logits.view(-1, logits.size(-1))
            targets = targets[:, 2:].contiguous().view(-1)  # Aligning targets
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, indices: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        """Generates new tokens given the input context."""
        for _ in range(max_new_tokens):
            if indices.size(1) < 3:
                context = torch.zeros((1, 3 - indices.size(1)), dtype=torch.long, device=device)
                indices = torch.cat((context, indices), dim=1)
            logits, _ = self(indices[:, -BLOCK_SIZE:])
            logits = logits[:, -1, :]
            probabilities = F.softmax(logits, dim=-1)
            next_index = torch.multinomial(probabilities, num_samples=1)
            indices = torch.cat((indices, next_index), dim=1)
        return indices

@torch.no_grad()
def estimate_loss(model, train_data, val_data):
    """Estimates loss for training and validation sets."""
    model.eval()
    losses = {'train': [], 'val': []}
    for split, data in [('train', train_data), ('val', val_data)]:
        split_losses = []
        for _ in range(EVALUATION_ITERATIONS):
            x, y = get_batch(data)
            _, loss = model(x, y)
            split_losses.append(loss.item())
        losses[split] = sum(split_losses) / len(split_losses)
    model.train()
    return losses

def train_model(model, train_data, val_data, model_path="trigram_model.pth"):
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    progress_bar = tqdm(range(MAX_ITERATIONS), desc="Training Progress", dynamic_ncols=True)

    for iteration in progress_bar:
        if iteration % EVALUATION_INTERVAL == 0:
            loss_values = estimate_loss(model, train_data, val_data)
            progress_bar.set_postfix({
                "Train Loss": f"{loss_values['train']:.4f}",
                "Val Loss": f"{loss_values['val']:.4f}"
            })

        x_batch, y_batch = get_batch(train_data)
        _, loss = model(x_batch, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    torch.save(model.state_dict(), model_path)
    print(f">>> Model saved to {model_path}")

def pretty_print_output(output_text: str):
    """Pretty prints generated text with a delay between characters."""
    for char in output_text:
        print(char, end='', flush=True)
        time.sleep(0.05)
    print()

def main():
    parser = argparse.ArgumentParser(description="Trigram Language Model")
    parser.add_argument('--mode', choices=['train', 'generate'], required=False, default='train', help="Mode: train or generate")
    parser.add_argument('--model_path', type=str, default='trigram_model.pth', help="Path to save/load model")
    args = parser.parse_args()
    
    text_data = load_text_data('data/input.txt')
    vocab_size, stoi, itos = create_vocab_mapping(text_data)
    encoded_text = torch.tensor(encode(text_data, stoi), dtype=torch.long)
    train_data, val_data = prepare_data(encoded_text)
    model = TrigramLanguageModel(vocab_size).to(device)
    
    if args.mode == 'train':
        train_model(model, train_data, val_data, args.model_path)
    elif args.mode == 'generate':
        print(f">>> Loading model from {args.model_path}")
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        model.eval()  # Ensure model is in evaluation mode
        context = torch.zeros((1, 1), dtype=torch.long, device=device)
        generated_indices = model.generate(context, max_new_tokens=500)[0].tolist()
        generated_text = decode(generated_indices, itos)
        pretty_print_output(generated_text)

if __name__ == "__main__":
    main()