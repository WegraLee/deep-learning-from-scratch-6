# 'A'의 경우
encoded = 'A'.encode("utf-8")
print(encoded)        # b'A'
print(list(encoded))  # [65]

# '가'의 경우
encoded = '가'.encode("utf-8")
print(encoded)        # b'\xea\xb0\x80'
print(list(encoded))  # [234, 160, 128]

ids = [65]
decoded = bytes(ids).decode("utf-8")
print(decoded)   # 'A'


class ByteTokenizer:
    def encode(self, text):
        return list(text.encode("utf-8"))

    def decode(self, ids):
        return bytes(ids).decode("utf-8")


# 사용 예
tokenizer = ByteTokenizer()
text = "hello월드😁"

# 인코딩
ids = tokenizer.encode(text)
print(ids)  # [104, 101, 108, 108, 111, 236, 155, 148, 235, 147, 156, 240, 159, 152, 129]

# 디코딩
decoded = tokenizer.decode(ids)
print(decoded)  # hello월드😁