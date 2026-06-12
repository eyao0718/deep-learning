# Define the vocabulary.
VOCAB = [
   "<sos>", "<eos>",
    "A", "B", "C", "D",
    "E", "F", "G", "H",
    "I", "J", "K", "L",
    "M", "N", "O", "P",
    "Q", "R", "S", "T",
    "U", "V", "W", "X",
    "Y", "Z", "'", " ", "<pad>"
]
# We have also included <sos> and <eos> in the vocabulary for you
# However in real life, you include it explicitly if not provided
VOCAB_MAP = {VOCAB[i]: i for i in range(0, len(VOCAB))}
PAD_TOKEN = VOCAB_MAP["<pad>"]
SOS_TOKEN = VOCAB_MAP["<sos>"]
EOS_TOKEN = VOCAB_MAP["<eos>"]

print(f"Length of Vocabulary    : {len(VOCAB)}")
print(f"VOCAB                   : {VOCAB}")
print(f"PAD_TOKEN               : {PAD_TOKEN}")
print(f"SOS_TOKEN               : {SOS_TOKEN}")
print(f"EOS_TOKEN               : {EOS_TOKEN}")

df_train = pd.read_csv("./dataset/train-clean-100/transcripts.csv")
df_val = pd.read_csv("./dataset/dev-clean/transcripts.csv")

df_train.head()