import datetime
import warnings

import torch
import torch.nn as nn
from torch.nn import functional as F
from codecarbon import EmissionsTracker
from tqdm.rich import trange
from tqdm import TqdmExperimentalWarning

warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)


############################### HYPERPARAMETERS ###############################

batch_size = 8  # how many independent sequences will we process in parallel?
max_iters = 10000
block_size = 512  # what is the maximum context length for predictions?
n_embd = 512
n_head = 8
n_layer = 12
learning_rate = 3e-4
dropout = 0.1
eval_interval = 100
eval_iters = 20
device = 'cuda' if torch.cuda.is_available() else 'cpu'
max_token_generation = 5000
train_corpus = "shakespeare"


################################## FUNCTIONS ##################################

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)   # (B,T,C)
        q = self.query(x) # (B,T,C)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * C**-0.5 # (B, T, C) @ (B, C, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,C)
        out = wei @ v # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

# super simple bigram model
class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in trange(max_new_tokens, desc="Generating tokens", leave=False):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

def generate_text(m, max_tokens: int, device: str, log_file, log_msg: str = "") -> str:
    """Generate a sample of text given the model, of size `max_tokens`."""
    m.eval()
    with torch.no_grad():
        context = torch.zeros((1, 1), dtype=torch.long, device=device)
        generation = decode(m.generate(context, max_new_tokens=max_tokens)[0].tolist())
    m.train()
    print(f"\n--- {log_msg} ---", file=log_file)
    print(generation, file=log_file)


################################### TRAINING ##################################

with EmissionsTracker(
    output_dir="/mnt/c/Users/Emile/.ecocarbonai/",
    project_name="nanoGPT",
    experiment_id="Shakespeare dataset - run 003",
    log_level="Error"
    ) as tracker:

    torch.manual_seed(1337)

    # wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
    with open(f"{train_corpus}/{train_corpus}.txt", 'r', encoding='utf-8') as f:
        text = f.read()

    # here are all the unique characters that occur in this text
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    # create a mapping from characters to integers
    stoi = { ch:i for i,ch in enumerate(chars) }
    itos = { i:ch for i,ch in enumerate(chars) }
    encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
    decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

    # Train and test splits
    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(0.9*len(data)) # first 90% will be train, rest val
    train_data = data[:n]
    val_data = data[n:]

    with open(f'{train_corpus}/log.txt', 'a') as f:
        print("--- NEW TRAINING STARTED (" + train_corpus.upper() +
            f") {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---",
            f"{batch_size = }",
            f"{max_iters = }",
            f"{block_size = }",
            f"{n_embd = }",
            f"{n_head = }",
            f"{n_layer = }",
            f"{learning_rate = }",
            f"{dropout = }\n",
            f"{eval_interval = }",
            f"{eval_iters = }",
            f"{device = }",
            f"{max_token_generation = }",
            sep='\n',
            file=f)
        model = BigramLanguageModel()
        m = model.to(device)
        best_val_loss = 100
        # print the number of parameters in the model
        print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters', file=f)

        # create a PyTorch optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

        for iter in trange(max_iters, desc="Looping through epochs"):

            # every once in a while evaluate the loss on train and val sets
            if iter % eval_interval == 0 or iter == max_iters - 1:
                log = (f"step {iter}: train_loss {losses['train']:.4f}, " + 
                      f"val_loss {losses['val']:.4f}")
                losses = estimate_loss()
                new_best = losses['val'] < best_val_loss
                if new_best:
                    torch.save(m.state_dict(), f"{train_corpus}/best_model.pt")
                    best_val_loss = losses['val']
                    log += f" -> {train_corpus}/best_model.pt"
                print(log, file=f)
                print(log)
                generate_text(m, max_tokens=max_token_generation // 10, device=device,
                              log_file=f, log_msg=f"EPOCH {iter}")

            # sample a batch of data
            xb, yb = get_batch('train')

            # evaluate the loss
            logits, loss = model(xb, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        # generate from the model
        print(f"Final text generation with {max_token_generation=}...")
        best_model = BigramLanguageModel().to(device)
        best_model.load_state_dict(torch.load(f"{train_corpus}/best_model.pt",
                                              weights_only=True))
        generate_text(m, max_tokens=max_token_generation, device=device, log_file=f,
                      log_msg="FINAL GENERATION")
        print("\n--- END OF TRAINING " +
              f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---\n\n",
              file=f)

###############################################################################