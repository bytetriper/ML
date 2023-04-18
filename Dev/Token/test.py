import tiktoken

enc=tiktoken.get_encoding('p50k_base')

s=enc.encode("I'm batman @ ")

print(enc.decode(s))