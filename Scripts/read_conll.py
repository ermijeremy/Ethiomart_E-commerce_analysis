def read_conll(file_path):
    sentences = []
    labels = []
    with open(file_path, encoding="utf-8") as f:
        tokens, tags = [], []
        for line in f:
            line = line.strip()
            if not line:
                if tokens:
                    sentences.append(tokens)
                    labels.append(tags)
                    tokens, tags = [], []
            else:
                splits = line.split("\t")
                if len(splits) == 2:
                    tokens.append(splits[0])
                    tags.append(splits[1])
    return sentences, labels
