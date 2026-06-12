# login to huggingface using token
login(token="")
# load tokenizer
if config["token_type"] == "1k":
    TOKENIZER = AutoTokenizer.from_pretrained("alexgichamba/hw4_tokenizer_1k")
    print("1k vocab tokenizer loaded")
elif config["token_type"] == "10k":
    TOKENIZER = AutoTokenizer.from_pretrained("alexgichamba/hw4_tokenizer_10k")
    print("10k vocab tokenizer loaded")
elif config["token_type"] == "char":
    TOKENIZER = CharTokenizer()
    print("character tokenizer loaded")
else:
    raise ValueError("Invalid token type")

UNK_TOKEN = TOKENIZER.unk_token_id
EOS_TOKEN = TOKENIZER.eos_token_id
SOS_TOKEN = TOKENIZER.bos_token_id
PAD_TOKEN = TOKENIZER.convert_tokens_to_ids("<|padding|>")