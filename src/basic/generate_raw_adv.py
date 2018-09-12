
# coding: utf-8

# In[1]:


from adversary import Adversary
from util import load_pickle, read_lines, write_lines
from argparse import ArgumentParser
from tqdm import tqdm
from os.path import join, exists


# In[ ]:


parser = ArgumentParser()
parser.add_argument(
    "strategy", type=str, help="strategy_name")
parser.add_argument(
    "--path", type=str, 
    default="data/",
    help="path to the Ubuntu Dialogue Corpus")
args = parser.parse_args()

path = args.path
strategy = args.strategy
print("Generating raw text data for {}".format(strategy))


# In[ ]:


dataset_dict_filename = "Dataset.dict.pkl"
raw_filenames = [
    "raw_training_text",
    "raw_valid_text",
    "raw_test_text"]
suffix = ".txt"


# In[ ]:


dataset_dict = load_pickle(join(path, dataset_dict_filename))
index2token = {index: token for (token, index, _, _) in dataset_dict}
token2index = {token: index for (token, index, _, _) in dataset_dict}
vocab = list(token2index.keys())
indices = list(token2index.values())

vocab_file = join(path, "vocab.txt")
if not exists(vocab_file):
    write_lines(vocab_file, vocab)


# In[ ]:


adversary = Adversary(vocab)


# In[ ]:


for filename in raw_filenames:
    file = join(path, "raw", filename + suffix)
    dialogues = read_lines(file)
    adv_dialogues = []
    for dialogue in tqdm(dialogues):
        adv_dialogue = adversary.apply_strategy(strategy, dialogue)
        adv_dialogues.append(adv_dialogue)    
    target_file = join(path, "raw", filename + "." + strategy + suffix)
    write_lines(target_file, adv_dialogues)

