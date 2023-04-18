with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read();

print("length of dataset in characters: ", len(text))

print("First 1000 characters: ")
print("------------START------------")
print(text[:1000])
print("-------------END-------------")

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

first_90pc_index = 0.9 * len(data)
train_data = data[:first_90pc_index]
valid_data = data[first_90pc_index:]
