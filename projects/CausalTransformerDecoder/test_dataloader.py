dl = DataLoaderForLanguageModeling(
    dataset = train_dataset,
    batch_size = config['batch_size'],
    shuffle = True,
    drop_last = False,
    sequence_length = 12,
)

# test dataloader
batch_max = 3
batch_cnt = 0
seq_max = 2
for x, y in dl:
    batch_cnt += 1
    print("")
    print("batch count:", batch_cnt)
    for seq in range(seq_max):
        print("")
        print("encoded x: ", x[seq])
        print("encoded y: ", y[seq])
        print("decoded x: ", TOKENIZER.decode(x[seq]))
        print("decoded y: ", TOKENIZER.decode(y[seq]))
    if batch_cnt >= batch_max:
        break