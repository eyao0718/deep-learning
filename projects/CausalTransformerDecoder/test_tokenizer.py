VOCAB_SIZE = TOKENIZER.vocab_size
print("vocab size:", VOCAB_SIZE)
print("")

# test the tokenizer
if TOKENIZER is not None:
    print("Testing decode for [EOS_TOKEN, SOS_TOKEN, PAD_TOKEN, UNK_TOKEN]...")
    print("Decoded result:", TOKENIZER.decode([EOS_TOKEN, SOS_TOKEN, PAD_TOKEN, UNK_TOKEN]))
    print("")
    print("Testing tokenizer for (HELLO DEEP LEARNERS)...")
    print("Tokenized result:", TOKENIZER.tokenize("HELLO DEEP LEARNERS"))
    print("Encoded result:", TOKENIZER.encode("HELLO DEEP LEARNERS"))
    print("")
    print(len(TOKENIZER.encode("CHAPTER TEN SHAGGY MAN TO THE RESCUE THEY HAD NOT GONE VERY FAR BEFORE BUNGLE WHO HAD RUN ON AHEAD CAME BOUNDING BACK TO SAY THAT THE ROAD OF YELLOW BRICKS WAS JUST BEFORE THEM")))