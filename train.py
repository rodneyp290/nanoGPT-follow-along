with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read();

print("length of dataset in characters: ", len(text))

print("First 1000 characters: ")
# print("------------START------------")
# print(text[:1000])
# print("-------------END-------------")

vocab_chars = sorted(list(set(text)))
vocab_size = len(vocab_chars)

print("Characters: \"" + "".join(vocab_chars) + "\"")
print("Number of chars:", vocab_size)

char_to_int = { ch:i for i,ch in enumerate(vocab_chars)}
int_to_char = { i:ch for i,ch in enumerate(vocab_chars)}

encode = lambda string: [char_to_int[c] for c in string]
decode = lambda ints: ''.join([int_to_char[i] for i in ints])

plain_text = "Hello There!"
encoded_text= encode(plain_text)
print(plain_text, "-> encode:", encoded_text, "-> decode:", decode(encoded_text))

import torch
data = torch.tensor(encode(text), dtype=torch.long)
print("data shape & type: ", data.shape, data.dtype)
print("data[:1000]:")
print("------------START------------")
print(data[:1000])
print("-------------END-------------")

first_90pc_index = int(0.9 * len(data))
train_data = data[:first_90pc_index]
valid_data = data[first_90pc_index:]

block_size = 8
train_data[:block_size+1]

x = train_data[:block_size]
y = train_data[1:block_size+1]
for t in range(block_size):
    context = x[:t+1]
    target = y[t]
    print(f"when context is {context} the target is: {target}")

torch.manual_seed(1337)
batch_size = 4
block_size = 8

def get_batch(split):
    data = train_data if split == 'train' else valid_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[1+i:1+i+block_size] for i in ix])
    return x, y

xb, yb = get_batch('train')
print(f"inputs: (shape: {xb.shape})")
print(xb)
print(f"targets: (shape: {yb.shape})")
print(yb)

print("------")

for b in range(batch_size): # batch dimension
    for t in range(block_size): # time dimension (? :S - sub block / context size?)
        context = xb[b,:t+1]
        target = yb[b, t]
        print(f"when input is context {context.tolist()} the target is: {target}")
