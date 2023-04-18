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
