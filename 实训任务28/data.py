
def gen_id2corpus(corpus_file):
    id2corpus = {}
    with open(corpus_file, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            id2corpus[idx] = line.rstrip()
    return id2corpus


def read_text_pair(data_path):
    """Reads data."""
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = line.rstrip().split('\t')
            # 无监督训练，text_a和text_b是一样的
            yield {'text_a': data[0], 'text_b': data[1]}

def convert_example_test(example,
                         tokenizer,
                         max_seq_length=512,
                         pad_to_max_seq_len=False):
    result = []
    for key, text in example.items():
        encoded_inputs = tokenizer(
            text=text,
            max_seq_len=max_seq_length,
            pad_to_max_seq_len=pad_to_max_seq_len)
        input_ids = encoded_inputs["input_ids"]
        token_type_ids = encoded_inputs["token_type_ids"]
        result += [input_ids, token_type_ids]
    return result

def get_rs(similar_text_pair,recall_result_file,recall_num):
    text2similar = {}
    with open(similar_text_pair, 'r', encoding='utf-8') as f:
        for line in f:
            text, similar_text = line.rstrip().split("\t")
            text2similar[text] = similar_text

    rs = []
    with open(recall_result_file, 'r', encoding='utf-8') as f:
        relevance_labels = []
        for index, line in enumerate(f):

            if index % recall_num == 0 and index != 0:
                rs.append(relevance_labels)
                relevance_labels = []
            text, recalled_text, cosine_sim = line.rstrip().split("\t")
            if text == recalled_text:
                continue
            if text2similar[text] == recalled_text:
                relevance_labels.append(1)
            else:
                relevance_labels.append(0) 
    return rs