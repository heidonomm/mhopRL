import torch
import nltk
from nltk import word_tokenize

missing_words = set()


def to_one_hot(y_true, n_classes):
    y_onehot = torch.FloatTensor(y_true.size(0), n_classes)
    if y_true.is_cuda:
        y_onehot = y_onehot.cuda()
    y_onehot.zero_()
    y_onehot.scatter_(1, y_true, 1)
    return y_onehot


def NegativeLogLoss(y_pred, y_true):
    """
    Shape:
        - y_pred:    batch x time
        - y_true:    batch
    """
    y_true_onehot = to_one_hot(y_true.unsqueeze(-1), y_pred.size(1))
    P = y_true_onehot.squeeze(-1) * y_pred  # batch x time
    P = torch.sum(P, dim=1)  # batch
    gt_zero = torch.gt(P, 0.0).float()  # batch
    epsilon = torch.le(P, 0.0).float() * 1e-8  # batch
    log_P = torch.log(P + epsilon) * gt_zero  # batch
    output = -log_P  # batch
    return output


def word_to_id(word, word2index):
    try:
        return word2index[word]
    except KeyError:
        key = word + "_" + str(len(word2index))
        if key not in missing_words:
            print("Warning... %s is not in vocab, vocab size is %d..." %
                  (word, len(word2index)))
            missing_words.add(key)
            with open("missing_words.txt", 'a+') as outfile:
                outfile.write(key + '\n')
                outfile.flush()
        return 1


def words_to_ids(text, word2index, convert_to_pt=False):
    ids = []
    tokens = word_tokenize(text)
    for token in tokens:
        ids.append(word_to_id(token, word2index))
    if convert_to_pt:
        ids_pt = torch.LongTensor(ids)
        return ids_pt
    return ids


def batch_words_to_idspt(batch, word2index):
    ids = []
    for sentence in batch:
        idx = words_to_ids(sentence, word2index, convert_to_pt=False)
        ids.append(idx)
    ids_pt = torch.LongTensor(ids)
    return ids_pt
