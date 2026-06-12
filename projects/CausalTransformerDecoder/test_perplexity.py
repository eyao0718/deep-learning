"""
Testing Your Model's Perplexity.

NOTE:
Perplexity score < 5 indicates model is good
and confident in making next token predictions.
"""



def log_softmax(x, axis):
    ret = x - np.max(x, axis=axis, keepdims=True)
    lsm = np.log(np.sum(np.exp(ret), axis=axis, keepdims=True))
    return ret - lsm

def get_prediction_nll_single_for_test(out, targ):
    out = log_softmax(out, 1)
    nlls = out[np.arange(out.shape[0]), targ]
    nll = -np.sum(nlls)
    return nll




test_df = pd.read_csv("./dataset/test-clean/transcripts.csv")
test_transcripts  = []
test_transcripts = [np.array([i for i in row['transcripts'].replace("<sos>", "").replace("<eos>", "") ]) for index, row in test_df.iterrows()]

test_dataset = []
for files in test_transcripts:
    tokenized = "".join(files)
    tokenized = TOKENIZER.encode(tokenized)
    test_dataset.append(tokenized)


test_dl = DataLoaderForLanguageModeling(
    dataset = test_dataset,
    batch_size = 1,
    shuffle = False,
    drop_last = False,
    sequence_length = 80,
)


nnls = []
model.eval()
model.to(DEVICE)
for batch_num, (inputs, targets) in enumerate(tqdm(test_dl)):
    inputs = torch.tensor(inputs).long().to(DEVICE)
    targets = torch.tensor(targets).long().to(DEVICE)

    with torch.no_grad():
        output = model(inputs)
    nnl = get_prediction_nll_single_for_test(output[0][0].to('cpu').numpy(), targets[0].to('cpu').numpy())
    if TOKENIZER != None:
        text_len = len(TOKENIZER.decode(targets.flatten().to('cpu')).replace("<|endoftext|>", "")) + 1
    else:
        text_len = len(targets[0])
    nnls.append(nnl / text_len)

test_ppl = np.exp(sum(nnls) / len(nnls))
print(f'test_perplexity: {test_ppl}')
with open('./test_perplexity.txt', 'w') as f:
    f.write(str(test_ppl))