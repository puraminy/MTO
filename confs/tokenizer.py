from transformers import T5Tokenizer

# Load the T5 tokenizer from the local directory
tokenizer = T5Tokenizer.from_pretrained("/home/ahmad/pret/t5-base")

# Input a word from the user
word = ""
while word != "end":
    word = input("Enter a word to tokenize: ")

    # Tokenize the word
    tokens = tokenizer.tokenize(word)
    token_ids = tokenizer.encode(word, add_special_tokens=False)

    # Print the results
    print(f"Word: '{word}'")
    print(f"Tokenized: {tokens}")
    print(f"Token IDs: {token_ids}")
