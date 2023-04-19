import torch 
import torch.nn as nn 
from torch.nn import functional as F

# hyperparameters
batch_size = 32 # how many independent sequences will we process in parallel
block_size = 8 # what is the maximum context length for prediction
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
device = "cuda" if torch.cuda.is_available() else "cpu"
eval_iters = 200
n_embed = 32
# ---------------

print(f"Running on {device}")

torch.manual_seed(1337)


with open("input.txt", "r", encoding="utf-8") as file:
    text = file.read()


# all uniq chars in text
chars = sorted(list(set(text)))
vocab_size = len(chars)

# mappings from chars to ints, and vice versa
char_to_int = {ch: i for i, ch in enumerate(chars)}
int_to_char = {i: ch for i, ch in enumerate(chars)}

# encode, and decode, to convert string to encoded integer list and back
encode = lambda string: [char_to_int[c] for c in string]
decode = lambda ints: "".join([int_to_char[i] for i in ints])

# Training and ~test~ ( validation) data split as tensors
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data)) # 90% mark index
train_data = data[:n]
val_data = data[n:]

# data loading (really? isn't it more data batch retrieving?)
def get_batch(split):
    # generate a small batch of data of input x and targets y
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[1+i:1+i+block_size] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# super simple bigram model
class BigramLanguageModel(nn.Module):
    
    def __init__(self):
        super().__init__()
        # each token directly reads odd the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, targets=None):
        
        # idx and targets are both (B,T) tensor of integers
        token_embed = self.token_embedding_table(idx) # (B,T,n_embed)
        logits = self.lm_head(token_embed) # (B,T,vocab_size)

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
        # i.e (Batch size, block size)
        for _ in range(max_new_tokens):
            # get predictions
            logits, _ = self(idx)
            # focus pm;y on the last step
            logits = logits[:, -1, :]  # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # sample from distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)

        return idx

model = BigramLanguageModel()
m = model.to(device)

# create a PyTorch Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
    
    # every once in a while evaluate the loss on the train and val set
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch("train")

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
