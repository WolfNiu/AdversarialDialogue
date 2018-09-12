
# coding: utf-8

# In[ ]:


from pprint import pprint
from tqdm import tqdm
from argparse import ArgumentParser
from util import read_lines, write_lines, dump_pickle
from os.path import join


# In[19]:


def _split(text, delimiter=None):
    text = text.strip()
    if text == "":
        return []
    else:
        splits = list(
            map(str.strip, text.split(delimiter)))
        if splits == []:
            return []
        elif splits[-1] == "":
            return splits[:(-1)]
        else:
            return splits


# In[12]:


class Indexer(object):
    def __init__(self,
                 vocab):
        self.vocab = vocab
        self.token2idx = {token: idx for idx, token in enumerate(vocab)}
        self.idx2token = {idx: token for idx, token in enumerate(vocab)}
        self.eot = "__eot__"
        self.unk_idx = vocab.index("**unknown**") # make sure **unknown** is the first token
    
    def _index_util(self, token):
        if token in self.token2idx:
            return self.token2idx[token]
        else:
            return self.unk_idx # corresponding to the **unknown** token
    
    def index(self, dialogue):
        # todo: optimize the two lines below [repeated code]
        tokenized_dialogue = [
            _split(turn) + [self.eot]
            for turn in _split(dialogue, delimiter=self.eot)]
        indexed_dialogue = [
            [self._index_util(token) 
             for token in turn]
            for turn in tokenized_dialogue]
        return indexed_dialogue
    
    def reverse_index(self, indexed_dialogue):
        if isinstance(indexed_dialogue[0], list):
            dialogue = [
                [self.idx2token[token] for token in turn]
                for turn in indexed_dialogue]
        elif isinstance(indexed_dialogue[0], int):
            dialogue = [
                self.idx2token[token] for token in indexed_dialogue]
        else:
            print("Something is wrong about the indexed dialogue.")
            raise
        return dialogue


# In[21]:


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "strategy", type=str, help="strategy_name")
    parser.add_argument(
        "--path", type=str, 
        default="data/",
        help="path to the Ubuntu Dialogue Corpus")
    parser.add_argument(
        "--bpe", action="store_true", help="whether using bpe")
    args = parser.parse_args()
    return args


# In[18]:


if __name__ == "__main__":    
    # Parse arguments
    args = parse_args()
    strategy = args.strategy
    path = args.path
    bpe = args.bpe

    # Get vocab and text
    source_path = join(path, "raw")
    target_path = join(path, "indexed")
    
    prefix = "bpe_" if bpe else ""
    vocab_file = join(path, f"{prefix}vocab.txt")
    vocab_lines = read_lines(vocab_file)
    vocab = [line.split()[0] for line in vocab_lines]       
    indexer = Indexer(vocab)
    
    splits = ["training", "valid", "test"]
    for split in splits:
        source_filename = f"{prefix}raw_{split}_text.{strategy}.txt"
        dialogues = read_lines(join(source_path, source_filename))
        indexed_dialogues = [
            indexer.index(dialogue) for dialogue in tqdm(dialogues)]
        target_filename = f"{prefix}indexed_{split}_text.{strategy}.pkl"
        dump_pickle(
            join(target_path, target_filename), 
            indexed_dialogues)    

