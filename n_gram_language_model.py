import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import time
from tqdm import tqdm

# Hyperparameters
torch.manual_seed(1337)
BATCH_SIZE = 32       # Number of sequences to process in parallel
BLOCK_SIZE = 32        # Context length for predictions
MAX_ITERATIONS = 25000
EVALUATION_INTERVAL = 500
LEARNING_RATE = 1e-3
EVALUATION_ITERATIONS = 200

def get_device():
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

def load_text_data(filepath):
    """Loads text data from a file."""
    print(f">>> Loading text data from {filepath}")
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()

def create_vocab_mapping(text):
    """Creates mappings between characters and indices."""
    characters = sorted(list(set(text)))
    vocab_size = len(characters)
    stoi = {char: idx for idx, char in enumerate(characters)}
    itos = {idx: char for idx, char in enumerate(characters)}
    return vocab_size, stoi, itos

def encode(text, stoi):
    """Encodes text into numerical format."""
    return [stoi[char] for char in text]

def decode(indices, itos):
    """Decodes numerical format back to text."""
    return ''.join([itos[i] for i in indices])

def prepare_data(encoded_text, train_ratio=0.85):
    """Splits data into training and validation sets."""
    split_index = int(train_ratio * len(encoded_text))
    return encoded_text[:split_index], encoded_text[split_index:]

def get_batch(data, n_gram):
    """Generates a batch of input-output pairs."""
    indices = torch.randint(len(data) - n_gram, (BATCH_SIZE,))
    x_batch = torch.stack([data[i:i + n_gram] for i in indices])
    y_batch = torch.stack([data[i + 1:i + n_gram + 1] for i in indices])
    return x_batch.to(device), y_batch.to(device)

class NGramLanguageModel(nn.Module):
    def __init__(self, vocab_size, n_gram):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, vocab_size)
        self.n_gram = n_gram

    def forward(self, indices, targets=None):
        """Forward pass through the model."""
        logits = self.embedding(indices)  # (B, T, C)
        loss = None
        if targets is not None:
            logits = logits.view(-1, logits.size(-1))  # Reshape for cross-entropy
            targets = targets.view(-1)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, indices, max_new_tokens):
        """Generates new tokens given input context."""
        for _ in range(max_new_tokens):
            if indices.size(1) < self.n_gram:
                pad_length = self.n_gram - indices.size(1)
                indices = torch.cat((torch.zeros((1, pad_length), dtype=torch.long, device=device), indices), dim=1)
            logits, _ = self(indices[:, -self.n_gram:])
            logits = logits[:, -1, :]  # Take the last time step
            probabilities = F.softmax(logits, dim=-1)
            next_index = torch.multinomial(probabilities, num_samples=1)
            indices = torch.cat((indices, next_index), dim=1)
        return indices

@torch.no_grad()
def estimate_loss(model, train_data, val_data, n_gram):
    """Estimates loss for training and validation sets."""
    model.eval()
    losses = {'train': [], 'val': []}
    for split, data in [('train', train_data), ('val', val_data)]:
        split_losses = []
        for _ in range(EVALUATION_ITERATIONS):
            x, y = get_batch(data, n_gram)
            _, loss = model(x, y)
            split_losses.append(loss.item())
        losses[split] = sum(split_losses) / len(split_losses)
    model.train()
    return losses

def train_model(model, train_data, val_data, n_gram, model_path):
    """Trains the model and saves weights."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    progress_bar = tqdm(range(MAX_ITERATIONS), desc="Training Progress", dynamic_ncols=True)

    for iteration in progress_bar:
        if iteration % EVALUATION_INTERVAL == 0:
            loss_values = estimate_loss(model, train_data, val_data, n_gram)
            progress_bar.set_postfix({
                "Train Loss": f"{loss_values['train']:.4f}",
                "Val Loss": f"{loss_values['val']:.4f}"
            })

        x_batch, y_batch = get_batch(train_data, n_gram)
        _, loss = model(x_batch, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    torch.save(model.state_dict(), model_path)
    print(f">>> Model saved to {model_path}")

def pretty_print_output(output_text):
    """Prints generated text with delay for readability."""
    for char in output_text:
        print(char, end='', flush=True)
        time.sleep(0.02)
    print()

def main():
    parser = argparse.ArgumentParser(description="N-Gram Language Model")
    parser.add_argument('--mode', choices=['train', 'generate'], required=True, help="Mode: train or generate")
    parser.add_argument('--n_gram', type=int, required=True, help="N-gram size")
    parser.add_argument('--model', type=str, required=False, help="Model file for generation")
    args = parser.parse_args()

    text_data = load_text_data('data/input.txt')
    vocab_size, stoi, itos = create_vocab_mapping(text_data)
    encoded_text = torch.tensor(encode(text_data, stoi), dtype=torch.long)
    train_data, val_data = prepare_data(encoded_text)

    model = NGramLanguageModel(vocab_size, args.n_gram).to(device)

    if args.mode == 'train':
        model_path = f"ngram_{args.n_gram}.pth"
        train_model(model, train_data, val_data, args.n_gram, model_path)
    elif args.mode == 'generate':
        if args.model is None:
            print(">>> Error: Must specify --model for generation")
            return
        print(f">>> Loading model from {args.model}")
        model.load_state_dict(torch.load(args.model, map_location=device))
        context = torch.zeros((1, 1), dtype=torch.long, device=device)
        generated_indices = model.generate(context, max_new_tokens=500)[0].tolist()
        generated_text = decode(generated_indices, itos)
        pretty_print_output(generated_text)

if __name__ == "__main__":
    main()