with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()

print("length of dataset in characters: ", len(text))

print("First 1000 characters: ")
# print("------------START------------")
# print(text[:1000])
# print("-------------END-------------")

vocab_chars = sorted(list(set(text)))
vocab_size = len(vocab_chars)

print('Characters: "' + "".join(vocab_chars) + '"')
print("Number of chars:", vocab_size)

char_to_int = {ch: i for i, ch in enumerate(vocab_chars)}
int_to_char = {i: ch for i, ch in enumerate(vocab_chars)}

encode = lambda string: [char_to_int[c] for c in string]
decode = lambda ints: "".join([int_to_char[i] for i in ints])

plain_text = "Hello There!"
encoded_text = encode(plain_text)
print(plain_text, "-> encode:", encoded_text, "-> decode:", decode(encoded_text))

import torch

data = torch.tensor(encode(text), dtype=torch.long)
# print("data shape & type: ", data.shape, data.dtype)
# print("data[:1000]:")
# print("------------START------------")
# print(data[:1000])
# print("-------------END-------------")

first_90pc_index = int(0.9 * len(data))
train_data = data[:first_90pc_index]
valid_data = data[first_90pc_index:]

block_size = 8
train_data[: block_size + 1]

x = train_data[:block_size]
y = train_data[1 : block_size + 1]
for t in range(block_size):
    context = x[: t + 1]
    target = y[t]
    print(f"when context is {context} the target is: {target}")

torch.manual_seed(1337)
batch_size = 4
block_size = 8


def get_batch(split):
    data = train_data if split == "train" else valid_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[1 + i : 1 + i + block_size] for i in ix])
    return x, y


xb, yb = get_batch("train")
print(f"inputs: (shape: {xb.shape})")
print(xb)
print(f"targets: (shape: {yb.shape})")
print(yb)

print("------")

for b in range(batch_size):  # batch dimension
    for t in range(block_size):  # time dimension (? :S - sub block / context size?)
        context = xb[b, : t + 1]
        target = yb[b, t]
        print(f"when input is context {context.tolist()} the target is: {target}")

print(xb)

# import torch # already done?
import torch.nn as nn
from torch.nn import functional as F

torch.manual_seed(1337)


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # each token directly reads off the logits(?) for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        # idx and targets are both (B,T) tensor of integers
        # ? B = batch_size? T = Time? (i.e. block_size?) (Yes)
        logits = self.token_embedding_table(idx)  # (B,T,C) #??

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
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


model = BigramLanguageModel(vocab_size)
logits, loss = model(xb, yb)
print("model output shape:", logits.shape)
print("loss: ", loss)
idx = torch.zeros((1, 1), dtype=torch.long)
generated = model.generate(idx, max_new_tokens=100)[0]
print(decode(generated.tolist()))

# create a PyTorch Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

batch_size = 32
for steps in range(int(5e5)):
    # sample a batch of data
    xb, yb = get_batch("train")

    #evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print(loss.item())
idx = torch.zeros((1, 1), dtype=torch.long)
generated = model.generate(idx, max_new_tokens=100)[0]
print(decode(generated.tolist()))