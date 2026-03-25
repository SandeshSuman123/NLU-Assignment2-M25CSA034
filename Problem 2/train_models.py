"""
CSL 7640: Natural Language Understanding — Assignment 2
Problem 2: Character-Level Name Generation using RNN Variants

Models implemented from scratch:
  1. Vanilla RNN
  2. Bidirectional LSTM (BLSTM)  
  3. RNN with Basic Attention Mechanism


"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

import numpy as np
import random
import os
import json
import matplotlib.pyplot as plt
from collections import Counter


# REPRODUCIBILITY

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

#HYPERPARAMETERS

HIDDEN_SIZE      = 256     # hidden state dimension (RNN & Attention)
HIDDEN_SIZE_LSTM = 128     # reduced for BLSTM to prevent overfitting
NUM_LAYERS       = 2
EMBEDDING_DIM    = 64
LEARNING_RATE    = 0.001
BATCH_SIZE       = 64
EPOCHS           = 30
MAX_NAME_LEN     = 20
NUM_GENERATE     = 200


DROPOUT_RNN      = 0.3    # dropout for Vanilla RNN and Attention
DROPOUT_LSTM     = 0.5    # higher dropout for BLSTM
LSTM_PATIENCE    = 8      # early stopping
LSTM_VAL_SPLIT   = 0.10   # 10% validation split for BLSTM only
TEMPERATURE_RNN  = 0.8    # sampling temperature for RNN & Attention
TEMPERATURE_LSTM = 1.0    # higher temperature for BLSTM 
MIN_NAME_LEN     = 3      # filter out generated names shorter than this

# Special tokens
PAD_TOKEN = '<PAD>'
SOS_TOKEN = '<SOS>'
EOS_TOKEN = '<EOS>'



# STEP 1: DATA LOADING & VOCABULARY


def load_names(filepath):
    """Read names from a text file (one name per line)."""
    with open(filepath, 'r', encoding='utf-8') as f:
        names = [line.strip() for line in f if line.strip()]
    # Normalise to title case so the model learns consistent capitalisation
    names = [n.title() for n in names]
    print(f"[Data] Loaded {len(names)} names from '{filepath}'")
    return names


class Vocabulary:
    """
    Maps individual characters to integer indices and back.
    Special tokens: <PAD>=0, <SOS>=1, <EOS>=2.
    """
    def __init__(self):
        self.char2idx = {PAD_TOKEN: 0, SOS_TOKEN: 1, EOS_TOKEN: 2}
        self.idx2char = {0: PAD_TOKEN, 1: SOS_TOKEN, 2: EOS_TOKEN}

    def build(self, names):
        # Collect every unique character across all names
        all_chars = sorted(set(''.join(names)))
        for ch in all_chars:
            if ch not in self.char2idx:
                idx = len(self.idx2char)   # use idx2char length (no duplicates)
                self.char2idx[ch]  = idx
                self.idx2char[idx] = ch
        print(f"[Vocab] Size = {len(self.idx2char)}  "
              f"(3 special + {len(all_chars)} unique chars)")

    def encode(self, name):
        """Convert a name string to a list of token indices."""
        return ([self.char2idx[SOS_TOKEN]]
                + [self.char2idx[c] for c in name]
                + [self.char2idx[EOS_TOKEN]])

    def decode(self, indices):
        """Convert token indices back to a name string (strips specials)."""
        chars = []
        for idx in indices:
            ch = self.idx2char.get(idx, '')
            if ch in (PAD_TOKEN, SOS_TOKEN, EOS_TOKEN):
                continue
            chars.append(ch)
        return ''.join(chars)

    def __len__(self):
        return len(self.idx2char)



# STEP 2: PYTORCH DATASET & DATALOADER


class NameDataset(Dataset):
    """
    Returns (input_seq, target_seq) pairs where:
      input_seq  = [SOS, c1, c2, ..., cn]
      target_seq = [c1,  c2, ..., cn, EOS]
    Both are padded to the same length within each batch via the collate fn.
    """
    def __init__(self, names, vocab):
        self.vocab   = vocab
        self.encoded = [vocab.encode(n) for n in names]

    def __len__(self):
        return len(self.encoded)

    def __getitem__(self, idx):
        seq = self.encoded[idx]
        # input : everything except the last token
        # target: everything except the first token
        return (torch.tensor(seq[:-1], dtype=torch.long),
                torch.tensor(seq[1:],  dtype=torch.long))


def collate_fn(batch):
    """Pad sequences inside a batch to the same length."""
    inputs, targets = zip(*batch)
    inputs  = nn.utils.rnn.pad_sequence(inputs,  batch_first=True, padding_value=0)
    targets = nn.utils.rnn.pad_sequence(targets, batch_first=True, padding_value=0)
    return inputs, targets



# STEP 3: MODEL DEFINITIONS


# 3a. Vanilla RNN 

class VanillaRNN(nn.Module):
    """
    Architecture:
      Embedding(vocab_size, 64)
        -> Dropout(0.3)
        -> RNN(64 -> 256, 2 layers, tanh, dropout=0.3)
        -> Dropout(0.3)
        -> Linear(256 -> vocab_size)

    Uses tanh activations (standard for vanilla RNN).
    Teacher-forcing is applied during training.

    Hyperparameters:
      embedding_dim : 64
      hidden_size   : 256
      num_layers    : 2
      dropout       : 0.3
      learning_rate : 0.001
    """
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers, dropout=0.3):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers  = num_layers

        # Character embedding table; padding_idx=0 means PAD embeddings are zero
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # Vanilla RNN with tanh non-linearity
        self.rnn = nn.RNN(
            input_size   = embedding_dim,
            hidden_size  = hidden_size,
            num_layers   = num_layers,
            batch_first  = True,
            dropout      = dropout if num_layers > 1 else 0.0,
            nonlinearity = 'tanh'
        )

        self.dropout = nn.Dropout(dropout)

        # Project hidden state to vocabulary logits
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        emb = self.dropout(self.embedding(x))       # (batch, seq, embed_dim)
        out, hidden = self.rnn(emb, hidden)         # (batch, seq, hidden)
        out    = self.dropout(out)
        logits = self.fc(out)                       # (batch, seq, vocab_size)
        return logits, hidden

    def init_hidden(self, batch_size, device):
        # h_0: (num_layers, batch, hidden_size)
        return torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


#3b. Bidirectional LSTM 

class BidirectionalLSTM(nn.Module):
    """
    Architecture:
      Embedding(vocab_size, 64)
        -> Dropout(0.5)
        -> BiLSTM(64 -> 128 per direction, 2 layers, dropout=0.5)
        -> Dropout(0.5)
        -> Linear(256 -> vocab_size)

    Forward pass: left-to-right.
    Backward pass: right-to-left.
    Output at each time-step: concatenation of both directions (2 * hidden_size).

    FIXES applied vs naive implementation:
      - hidden_size reduced 256 -> 128  (smaller capacity reduces memorisation)
      - dropout increased 0.3 -> 0.5    (stronger regularisation)
      - trained with early stopping on validation loss (see train_blstm())
      - generation uses MIN_NAME_LEN filter to remove single-char outputs

    Hyperparameters:
      embedding_dim : 64
      hidden_size   : 128  (per direction; output dim = 256)
      num_layers    : 2
      dropout       : 0.5
      learning_rate : 0.001
    """
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers, dropout=0.5):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers  = num_layers

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # bidirectional=True -> output dim per step = 2 * hidden_size
        self.lstm = nn.LSTM(
            input_size    = embedding_dim,
            hidden_size   = hidden_size,
            num_layers    = num_layers,
            batch_first   = True,
            bidirectional = True,
            dropout       = dropout if num_layers > 1 else 0.0
        )

        self.dropout = nn.Dropout(dropout)

        # 2 * hidden_size because fwd and bwd hidden states are concatenated
        self.fc = nn.Linear(2 * hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        emb = self.dropout(self.embedding(x))       # (batch, seq, embed)
        out, hidden = self.lstm(emb, hidden)        # (batch, seq, 2*hidden)
        out    = self.dropout(out)
        logits = self.fc(out)                       # (batch, seq, vocab_size)
        return logits, hidden

    def init_hidden(self, batch_size, device):
        # LSTM needs h_0 and c_0; factor of 2 for bidirectional
        h = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size, device=device)
        c = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size, device=device)
        return (h, c)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# 3c. RNN with Basic Attention Mechanism

class Attention(nn.Module):
    """
    Additive (Bahdanau-style) attention.

    score(query, key) = v^T . tanh(W_q . query + W_k . key)

    The attention weights form a probability distribution over all previous
    hidden states; the context vector is their weighted sum.
    """
    def __init__(self, hidden_size):
        super().__init__()
        self.W_query = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_key   = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v       = nn.Linear(hidden_size, 1,           bias=False)

    def forward(self, query, keys):
        """
        query : (batch, 1, hidden_size)
        keys  : (batch, seq_len, hidden_size)
        Returns context (batch, 1, hidden_size) and weights (batch, 1, seq_len).
        """
        q       = self.W_query(query)                        # (batch, 1, hidden)
        k       = self.W_key(keys)                           # (batch, seq, hidden)
        energy  = self.v(torch.tanh(q + k))                  # (batch, seq, 1)
        weights = F.softmax(energy, dim=1)                   # normalise over seq
        context = torch.bmm(weights.transpose(1, 2), keys)  # (batch, 1, hidden)
        return context, weights.transpose(1, 2)


class RNNWithAttention(nn.Module):
    """
    Architecture:
      Embedding(vocab_size, 64)
        -> Dropout(0.3)
        -> RNN(64 -> 256, 2 layers, tanh, dropout=0.3)
        -> Attention(query=last hidden, keys=all previous hidden states)
        -> concat(hidden_t, context)            # (batch, seq, 512)
        -> Dropout(0.3)
        -> Linear(512 -> vocab_size)

    The memory tensor accumulates all hidden states produced so far, allowing
    the attention mechanism to focus on any previous character in the name.

    Hyperparameters:
      embedding_dim : 64
      hidden_size   : 256
      num_layers    : 2
      dropout       : 0.3
      learning_rate : 0.001
    """
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers, dropout=0.3):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers  = num_layers

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        self.rnn = nn.RNN(
            input_size   = embedding_dim,
            hidden_size  = hidden_size,
            num_layers   = num_layers,
            batch_first  = True,
            dropout      = dropout if num_layers > 1 else 0.0,
            nonlinearity = 'tanh'
        )

        # Attention over accumulated hidden states
        self.attention = Attention(hidden_size)
        self.dropout   = nn.Dropout(dropout)

        # hidden_size (current) + hidden_size (context) -> vocab_size
        self.fc = nn.Linear(hidden_size * 2, vocab_size)

    def forward(self, x, hidden=None, memory=None):
        """
        x      : (batch, seq_len)
        hidden : RNN hidden state  (num_layers, batch, hidden)
        memory : accumulated hidden states  (batch, t, hidden) or None
        Returns logits, new hidden, updated memory.
        """
        emb = self.dropout(self.embedding(x))       # (batch, seq, embed)
        out, hidden = self.rnn(emb, hidden)         # (batch, seq, hidden)

        # Accumulate hidden states for attention memory
        memory = out if memory is None else torch.cat([memory, out], dim=1)

        # Use last time-step output as the attention query
        query      = out[:, -1:, :]                         # (batch, 1, hidden)
        context, _ = self.attention(query, memory)          # (batch, 1, hidden)

        # Expand context across all time-steps in this chunk
        context_exp = context.expand(-1, out.size(1), -1)  # (batch, seq, hidden)
        combined    = torch.cat([out, context_exp], dim=-1) # (batch, seq, 2*hidden)
        combined    = self.dropout(combined)
        logits      = self.fc(combined)                     # (batch, seq, vocab_size)
        return logits, hidden, memory

    def init_hidden(self, batch_size, device):
        return torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)



# STEP 4: TRAINING


def train_epoch(model, loader, optimizer, criterion, device, model_type):
    """One epoch of teacher-forced training. Returns average loss."""
    model.train()
    total_loss = 0.0

    for inputs, targets in loader:
        inputs     = inputs.to(device)
        targets    = targets.to(device)
        batch_size = inputs.size(0)
        optimizer.zero_grad()

        if model_type == 'attention':
            logits, _, _ = model(inputs, model.init_hidden(batch_size, device))
        else:
            logits, _    = model(inputs, model.init_hidden(batch_size, device))

        # Reshape for cross-entropy: (batch*seq, vocab) vs (batch*seq,)
        loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        loss.backward()

        # Gradient clipping prevents exploding gradients in RNNs
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


def train_model(model, dataset, epochs, device, model_type, model_name):
    """
    Standard training loop for Vanilla RNN and Attention RNN.
    Returns list of per-epoch train losses.
    """
    loader    = DataLoader(dataset, batch_size=BATCH_SIZE,
                           shuffle=True, collate_fn=collate_fn)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    losses = []
    print(f"\n{'='*60}")
    print(f"  Training: {model_name}  |  params: {model.count_parameters():,}")
    print(f"{'='*60}")

    for epoch in range(1, epochs + 1):
        loss = train_epoch(model, loader, optimizer, criterion, device, model_type)
        scheduler.step(loss)
        losses.append(loss)
        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:>3}/{epochs}  |  Loss: {loss:.4f}  "
                  f"|  LR: {optimizer.param_groups[0]['lr']:.6f}")

    return losses


def train_blstm(model, full_dataset, epochs, device):
    """
    BLSTM-specific training loop with:
      - 90/10 train/val split to monitor generalisation
      - ReduceLROnPlateau scheduler
      - Early stopping (stops when val loss stagnates for LSTM_PATIENCE epochs)
      - Best weights restored at the end

    Returns train_losses, val_losses, best_epoch.
    """
    # Split dataset into train and validation
    val_size   = int(len(full_dataset) * LSTM_VAL_SPLIT)
    train_size = len(full_dataset) - val_size
    train_ds, val_ds = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(SEED)
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True,  collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                              shuffle=False, collate_fn=collate_fn)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    best_state    = None
    best_epoch    = 1
    no_improve    = 0

    print(f"\n{'='*60}")
    print(f"  Training: Bidirectional LSTM (Fixed)  |  "
          f"params: {model.count_parameters():,}")
    print(f"  hidden={HIDDEN_SIZE_LSTM}, dropout={DROPOUT_LSTM}, "
          f"patience={LSTM_PATIENCE}")
    print(f"{'='*60}")

    for epoch in range(1, epochs + 1):

        # Training pass
        model.train()
        t_loss = 0.0
        for inputs, targets in train_loader:
            inputs  = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            logits, _ = model(inputs, model.init_hidden(inputs.size(0), device))
            loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            t_loss += loss.item()
        t_loss /= len(train_loader)

        # Validation pass
        model.eval()
        v_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs  = inputs.to(device)
                targets = targets.to(device)
                logits, _ = model(inputs, model.init_hidden(inputs.size(0), device))
                loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
                v_loss += loss.item()
        v_loss /= len(val_loader)

        scheduler.step(v_loss)
        train_losses.append(t_loss)
        val_losses.append(v_loss)

        # Save best weights
        if v_loss < best_val_loss:
            best_val_loss = v_loss
            best_state    = {k: v.clone() for k, v in model.state_dict().items()}
            best_epoch    = epoch
            no_improve    = 0
            marker        = "  <- best"
        else:
            no_improve += 1
            marker      = ""

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:>3}/{epochs}  "
                  f"train={t_loss:.4f}  val={v_loss:.4f}  "
                  f"lr={optimizer.param_groups[0]['lr']:.6f}{marker}")

        # Early stopping
        if no_improve >= LSTM_PATIENCE:
            print(f"\n  [Early Stop] No val improvement for {LSTM_PATIENCE} "
                  f"epochs. Stopping at epoch {epoch}.")
            break

    # Restore the best checkpoint
    model.load_state_dict(best_state)
    print(f"\n  [Best] Epoch {best_epoch}  val_loss={best_val_loss:.4f}")
    return train_losses, val_losses, best_epoch



# STEP 5: NAME GENERATION (AUTOREGRESSIVE SAMPLING)


def generate_name_rnn(model, vocab, device, temperature=TEMPERATURE_RNN):
    """Temperature sampling for Vanilla RNN."""
    model.eval()
    with torch.no_grad():
        hidden    = model.init_hidden(1, device)
        inp       = torch.tensor([[vocab.char2idx[SOS_TOKEN]]], device=device)
        generated = []
        for _ in range(MAX_NAME_LEN):
            logits, hidden = model(inp, hidden)
            probs    = F.softmax(logits.squeeze() / temperature, dim=-1)
            next_idx = torch.multinomial(probs, num_samples=1).item()
            if next_idx == vocab.char2idx[EOS_TOKEN]: break
            if next_idx == vocab.char2idx[PAD_TOKEN]: continue
            generated.append(next_idx)
            inp = torch.tensor([[next_idx]], device=device)
    return vocab.decode(generated)


def generate_name_blstm(model, vocab, device, temperature=TEMPERATURE_LSTM):
    """Temperature sampling for BLSTM."""
    model.eval()
    with torch.no_grad():
        hidden    = model.init_hidden(1, device)
        inp       = torch.tensor([[vocab.char2idx[SOS_TOKEN]]], device=device)
        generated = []
        for _ in range(MAX_NAME_LEN):
            logits, hidden = model(inp, hidden)
            probs    = F.softmax(logits.squeeze() / temperature, dim=-1)
            next_idx = torch.multinomial(probs, num_samples=1).item()
            if next_idx == vocab.char2idx[EOS_TOKEN]: break
            if next_idx == vocab.char2idx[PAD_TOKEN]: continue
            generated.append(next_idx)
            inp = torch.tensor([[next_idx]], device=device)
    return vocab.decode(generated)


def generate_name_attention(model, vocab, device, temperature=TEMPERATURE_RNN):
    """Temperature sampling for RNN with Attention."""
    model.eval()
    with torch.no_grad():
        hidden    = model.init_hidden(1, device)
        memory    = None
        inp       = torch.tensor([[vocab.char2idx[SOS_TOKEN]]], device=device)
        generated = []
        for _ in range(MAX_NAME_LEN):
            logits, hidden, memory = model(inp, hidden, memory)
            probs    = F.softmax(logits.squeeze() / temperature, dim=-1)
            next_idx = torch.multinomial(probs, num_samples=1).item()
            if next_idx == vocab.char2idx[EOS_TOKEN]: break
            if next_idx == vocab.char2idx[PAD_TOKEN]: continue
            generated.append(next_idx)
            inp = torch.tensor([[next_idx]], device=device)
    return vocab.decode(generated)


def generate_names(model, vocab, device, model_type, n=NUM_GENERATE):
    """
    Generate n names. For BLSTM, applies MIN_NAME_LEN filter to avoid
    short/meaningless outputs like 'O', 'Z', 'Im'.
    """
    generate_fn = {
        'rnn':       generate_name_rnn,
        'blstm':     generate_name_blstm,
        'attention': generate_name_attention,
    }[model_type]

    names    = []
    attempts = 0
    max_att  = n * 10   # safety cap to avoid infinite loop

    while len(names) < n and attempts < max_att:
        name = generate_fn(model, vocab, device)
        attempts += 1
        # BLSTM: enforce minimum length to remove degenerate short outputs
        if model_type == 'blstm' and len(name) < MIN_NAME_LEN:
            continue
        if name:
            names.append(name)

    if model_type == 'blstm':
        print(f"  [BLSTM] Generated {len(names)} valid names "
              f"in {attempts} attempts (min_len={MIN_NAME_LEN})")
    return names



# STEP 6: EVALUATION METRICS


def compute_metrics(generated, training_names):
    """
    Novelty Rate : % of generated names NOT in the training set.
    Diversity    : unique generated names / total generated names * 100.
    """
    training_set = set(n.lower() for n in training_names)
    total        = len(generated)
    unique       = set(n.lower() for n in generated)
    novel        = sum(1 for n in generated if n.lower() not in training_set)
    return {
        'total_generated': total,
        'unique_names':    len(unique),
        'novelty_rate':    round(novel / total * 100, 2) if total else 0.0,
        'diversity':       round(len(unique) / total * 100, 2) if total else 0.0,
    }



# STEP 7: VISUALISATION

def plot_losses(loss_rnn, loss_lstm_train, loss_lstm_val,
                loss_attn, best_epoch, save_path='training_losses.png'):
    """
    Combined loss plot:
      - Vanilla RNN and Attention: single train curve each
      - BLSTM: both train and val curves + best-epoch marker
    A close train/val gap for BLSTM confirms overfitting is fixed.
    """
    epochs_rnn  = range(1, len(loss_rnn)        + 1)
    epochs_lstm = range(1, len(loss_lstm_train) + 1)
    epochs_attn = range(1, len(loss_attn)       + 1)

    plt.figure(figsize=(11, 5))
    plt.plot(epochs_rnn,  loss_rnn,        color='#e74c3c', linewidth=2,
             label='Vanilla RNN (train)')
    plt.plot(epochs_lstm, loss_lstm_train, color='#3498db', linewidth=2,
             label='BLSTM (train)')
    plt.plot(epochs_lstm, loss_lstm_val,   color='#3498db', linewidth=2,
             linestyle='--', label='BLSTM (val)')
    plt.plot(epochs_attn, loss_attn,       color='#2ecc71', linewidth=2,
             label='RNN + Attention (train)')
    plt.axvline(x=best_epoch, color='#3498db', linestyle=':',
                linewidth=1.5, label=f'BLSTM best epoch ({best_epoch})')

    plt.xlabel('Epoch', fontsize=13)
    plt.ylabel('Cross-Entropy Loss', fontsize=13)
    plt.title('Training Loss Curves — Character-Level Name Generation', fontsize=14)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[Plot] Saved -> '{save_path}'")


def print_architecture_summary(model, model_name, hidden, dropout):
    """Print a readable architecture and hyperparameter summary."""
    print(f"\n{'─'*60}")
    print(f"  Architecture: {model_name}")
    print(f"{'─'*60}")
    print(model)
    print(f"\n  Trainable parameters : {model.count_parameters():,}")
    print(f"  embedding_dim        : {EMBEDDING_DIM}")
    print(f"  hidden_size          : {hidden}")
    print(f"  num_layers           : {NUM_LAYERS}")
    print(f"  dropout              : {dropout}")
    print(f"  learning_rate        : {LEARNING_RATE}")
    print(f"  batch_size           : {BATCH_SIZE}")
    print(f"  epochs               : {EPOCHS}")


# MAIN PIPELINE


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Device] Using: {device}")

    # Load data
    names = load_names('TrainingNames.txt')

    # Build vocabulary
    vocab      = Vocabulary()
    vocab.build(names)
    vocab_size = len(vocab)

    # Dataset
    dataset = NameDataset(names, vocab)

    # Instantiate models
    rnn_model = VanillaRNN(
        vocab_size    = vocab_size,
        embedding_dim = EMBEDDING_DIM,
        hidden_size   = HIDDEN_SIZE,
        num_layers    = NUM_LAYERS,
        dropout       = DROPOUT_RNN,
    ).to(device)

    # BLSTM: reduced hidden size + higher dropout to fix overfitting
    blstm_model = BidirectionalLSTM(
        vocab_size    = vocab_size,
        embedding_dim = EMBEDDING_DIM,
        hidden_size   = HIDDEN_SIZE_LSTM,   # 128 (was 256)
        num_layers    = NUM_LAYERS,
        dropout       = DROPOUT_LSTM,       # 0.5 (was 0.3)
    ).to(device)

    attn_model = RNNWithAttention(
        vocab_size    = vocab_size,
        embedding_dim = EMBEDDING_DIM,
        hidden_size   = HIDDEN_SIZE,
        num_layers    = NUM_LAYERS,
        dropout       = DROPOUT_RNN,
    ).to(device)

    # Print architecture summaries
    print_architecture_summary(rnn_model,   "Vanilla RNN",                HIDDEN_SIZE,      DROPOUT_RNN)
    print_architecture_summary(blstm_model, "Bidirectional LSTM (Fixed)", HIDDEN_SIZE_LSTM, DROPOUT_LSTM)
    print_architecture_summary(attn_model,  "RNN with Attention",         HIDDEN_SIZE,      DROPOUT_RNN)

    # Train RNN and Attention with standard loop
    loss_rnn  = train_model(rnn_model,  dataset, EPOCHS, device, 'rnn',       "Vanilla RNN")
    loss_attn = train_model(attn_model, dataset, EPOCHS, device, 'attention', "RNN + Attention")

    # Train BLSTM with early stopping + val monitoring
    loss_lstm_train, loss_lstm_val, best_epoch = train_blstm(
        blstm_model, dataset, EPOCHS, device)

    # Plot all losses on one figure
    plot_losses(loss_rnn, loss_lstm_train, loss_lstm_val, loss_attn, best_epoch)

    # Generate names
    print(f"\n[Generate] Sampling {NUM_GENERATE} names from each model ...")
    gen_rnn  = generate_names(rnn_model,   vocab, device, 'rnn')
    gen_lstm = generate_names(blstm_model, vocab, device, 'blstm')
    gen_attn = generate_names(attn_model,  vocab, device, 'attention')

    # Evaluate
    metrics_rnn  = compute_metrics(gen_rnn,  names)
    metrics_lstm = compute_metrics(gen_lstm, names)
    metrics_attn = compute_metrics(gen_attn, names)

    print("\n" + "="*62)
    print("  EVALUATION RESULTS")
    print("="*62)
    print(f"  {'Model':<25} {'Total':>6} {'Unique':>7} {'Novelty%':>10} {'Diversity%':>11}")
    print("  " + "─" * 58)
    for mname, m in [("Vanilla RNN",        metrics_rnn),
                     ("Bidirectional LSTM",  metrics_lstm),
                     ("RNN + Attention",     metrics_attn)]:
        print(f"  {mname:<25} {m['total_generated']:>6} {m['unique_names']:>7} "
              f"{m['novelty_rate']:>9.2f}% {m['diversity']:>10.2f}%")

    # Sample generated names
    print("\n" + "="*62)
    print("  SAMPLE GENERATED NAMES (first 20 each)")
    print("="*62)
    for mname, gen in [("Vanilla RNN",       gen_rnn[:20]),
                       ("Bidirectional LSTM", gen_lstm[:20]),
                       ("RNN + Attention",    gen_attn[:20])]:
        print(f"\n  [{mname}]")
        print("  " + ",  ".join(gen))

    # Save generated names
    for fname, gen in [("generated_rnn.txt",       gen_rnn),
                       ("generated_blstm.txt",      gen_lstm),
                       ("generated_attention.txt",  gen_attn)]:
        with open(fname, 'w') as f:
            f.write('\n'.join(gen))
        print(f"[Save] '{fname}'")

    # Save model checkpoints
    torch.save(rnn_model.state_dict(),   'checkpoint_rnn.pt')
    torch.save(blstm_model.state_dict(), 'checkpoint_blstm.pt')
    torch.save(attn_model.state_dict(),  'checkpoint_attention.pt')
    print("[Save] Model checkpoints saved.")

    # Save metrics as JSON
    results = {
        'Vanilla RNN':        metrics_rnn,
        'Bidirectional LSTM': metrics_lstm,
        'RNN + Attention':    metrics_attn,
    }
    with open('evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("[Save] 'evaluation_results.json'")


if __name__ == '__main__':
    main()