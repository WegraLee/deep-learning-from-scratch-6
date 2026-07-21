text = "hello월드😁"
print(list(text))  # ['h', 'e', 'l', 'l', 'o', '월', '드', '😁']

print(ord('h'))   # 104
print(ord('😁'))  # 128513

print(chr(104))    # 'h'
print(chr(128513)) # '😁'

ids = [ord(char) for char in list(text)]
print(ids)  # [104, 101, 108, 108, 111, 50900, 46300, 128513]

class CharTokenizer:
    def encode(self, text):
        return [ord(char) for char in text]

    def decode(self, ids):
        return ''.join([chr(i) for i in ids])

tokenizer = CharTokenizer()
text = "hello월드😁"

# 인코딩
ids = tokenizer.encode(text)
print(ids)  # [104, 101, 108, 108, 111, 50900, 46300, 128513]

# 디코딩
decoded = tokenizer.decode(ids)
print(decoded)  # hello월드😁