# list of arrays, each array contains characters for single row
train_transcripts = [np.array([char for char in row['transcripts'].replace("<sos>", "").replace("<eos>", "")]) for index, row in df_train.iterrows()]
train_dataset = []
for files in train_transcripts:
    tokenized = "".join(files)
    # tokenized: list of id
    tokenized = TOKENIZER.encode(tokenized)
    train_dataset.append(tokenized)

# list of arrays, each array contains characters for single row
val_transcripts = [np.array([i for i in row['transcripts'].replace("<sos>", "").replace("<eos>", "")]) for index, row in df_val.iterrows()]
val_dataset = []
for files in val_transcripts:
    tokenized = "".join(files)
    # tokenized: list of id
    tokenized = TOKENIZER.encode(tokenized)
    val_dataset.append(tokenized)

print("train dataset:", len(train_dataset))
print("val dataset:", len(val_dataset))