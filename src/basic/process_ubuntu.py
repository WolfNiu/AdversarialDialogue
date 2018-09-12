
# coding: utf-8

# In[ ]:



# coding: utf-8

# In[25]:


"""
Todo!!
1. " ".join each utterance
2. Serve turns in order within one dialogue
    (like in LM)
3. Never filter out examples with long sequence, otherwise turn progression will be lost.
4. Remember to shuffle all the dialogues first
5. When training, no need to modfiy "previous_context_input", 
    just set the "start_turn_index" correctly!

Todo: combine read_lines, load_pickle, etc... to one single function load_file(),
    and use if statement to see which suffix the file has. Also keep an optional param
    suffix=None just incase we want to force it to load with a certain format
"""

import numpy as np
import itertools
from pprint import pprint
import random
from nltk import FreqDist
from nltk.tokenize import word_tokenize
from nltk.tag.stanford import StanfordPOSTagger
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from nltk import pos_tag
from src.basic.util import load_pickle, read_lines, zip_lsts, split_lst, unzip_lst
import math
import string
import copy
from pathlib import Path
from pattern import en
from functools import reduce
from nltk.corpus import wordnet


# In[26]:


path = "/playpen/home/tongn/processed_ubuntu/"
dataset_dict_file = "Dataset.dict.pkl"
dataset_dict = load_pickle(path + dataset_dict_file)
index2token = {index: token for (token, index, _, _) in dataset_dict}
token2index = {token: index for (token, index, _, _) in dataset_dict}
vocab = list(token2index.keys())
indices = list(token2index.values())

eot_token = "__eot__"
eou_token = "__eou__"
unk_token = "**unknown**"

eot = token2index[eot_token]
eou = token2index[eou_token]
unk = token2index[unk_token]

global num_replaces
num_replaces = 0


# In[15]:


def to_str_lst(indices):
    str_lst = [index2token[index] for index in indices]
    return str_lst

def index(token):
    try:
        return token2index[token]
    except:
        return unk

def double_trans(translator, utterance):
    try:
        utterance_str = ' '.join([index2token[index] for index in utterance])
    #     print(utterance_str) 
        utterance_de = translator.translate(utterance_str, dest="de", src="en").text
        utterance_adversarial = translator.translate(utterance_de, dest="en", src="de").text.lower()
    #     print(utterance_adversarial)
        # correct a translation "error" where **unknown** is separated
        utterance_adversarial = utterance_adversarial.replace("** unknown **", "**unknown**")
    #     print(utterance_adversarial)
    #     print(" ".join(word_tokenize(utterance_adversarial)))
        indexed_adversarial = list(map(index, word_tokenize(utterance_adversarial.lower())))
        return indexed_adversarial
    except:
        print("Warning: unsuccessful with utterance:", utterance)
        return utterance

para_dict = load_pickle("/playpen/home/tongn/pointer_generator_log/ubuntu/para_dict.pkl")
    
def get_para(utterance):
    try:
        paraphrased_utterance = para_dict[tuple(utterance)]
        return paraphrased_utterance + [eou]
    except:
        return utterance + [eou]
    
def generative_paraphrase(utterance):
    """strategy 7"""
    turns = split_lst(utterance[:(-1)], eou, keep_delimiter=False)
    paraphrased_utterances = list(map(get_para, turns))
    paraphrased_utterances_concat = list(itertools.chain(*paraphrased_utterances))
    turns_concat = paraphrased_utterances_concat + [eot]
    return turns_concat
    
# def generative_paraphrase(utterance):
#     """strategy 7"""
# #     translator = Translator()
# #     turns = [
# #         [double_trans(translator, utterance[:(-1)]) for utterance in split_lst(turn[:(-1)], eou)]
# #         for turn in split_lst(utterance, eot)]
# #     utterances_concat = [
# #         reduce(lambda a, b: (a + [eou] + b), turn) + [eou] for turn in turns]
# #     turns_concat = reduce(lambda a, b: (a + [eot] + b), utterances_concat)
#     turns_concat = utterance # when testing with strategy 7, we already have parphrase.Test.dialogues.pkl
    
#     return turns_concat


# In[16]:


# x = [1539, 4745, 10269, 17925, 6916, 8070, 19766, 9267, 18828, eou, 
#      0, 4563, 8828, 11680, 7336, 19883, 5971, 1968, 780, eou,
#      eot]
# generative_paraphrase(x)


# In[5]:


# utterance_str = "im trying to use ubuntu on my macbook pro **unknown** __eou__ i read in the forums that ubuntu has apple version now ? __eou__ __eot__ not that ive ever heard of .. normal ubutnu should work on an intel based mac . there is PPC version also . __eou__ you want total control ? or what are you wanting exactly ? __eou__"
# utterance = list(map(lambda x: token2index[x], utterance_str.split()))
# paraphrased_utterance = generative_paraphrase(utterance)

# print(paraphrased_utterance)
# print(utterance_str)
# print(' '.join(list(map(lambda x: index2token[x], paraphrased_utterance))))


# In[23]:


"""
• POS tags we need:
    JJ	adjective	green
    JJR	adjective, comparative	greener
    JJS	adjective, superlative	greenest
    
VB: verb, base form
    ask assemble assess assign assume atone attention avoid bake balkanize
    bank begin behold believe bend benefit bevel beware bless boil bomb
    boost brace break bring broil brush build ...
VBD: verb, past tense
    dipped pleaded swiped regummed soaked tidied convened halted registered
    cushioned exacted snubbed strode aimed adopted belied figgered
    speculated wore appreciated contemplated ...
VBG: verb, present participle or gerund
    telegraphing stirring focusing angering judging stalling lactating
    hankerin' alleging veering capping approaching traveling besieging
    encrypting interrupting erasing wincing ...
VBN: verb, past participle
    multihulled dilapidated aerosolized chaired languished panelized used
    experimented flourished imitated reunifed factored condensed sheared
    unsettled primed dubbed desired ...
VBP: verb, present tense, not 3rd person singular
    predominate wrap resort sue twist spill cure lengthen brush terminate
    appear tend stray glisten obtain comprise detest tease attract
    emphasize mold postpone sever return wag ...
VBZ: verb, present tense, 3rd person singular
    bases reconstructs marks mixes displeases seals carps weaves snatches
    slumps stretches authorizes smolders pictures emerges stockpiles
    seduces fizzes uses bolsters slaps speaks pleads ...
    
    VB	verb be, base form	be
        VV	verb, base form	take
    VBD	verb be, past tense	was, were
        VVD	verb, past tense	took
    VBG	verb be, gerund/present participle	being
        VVG	verb, gerund/present participle	taking
    VBN	verb be, past participle	been
        VVN	verb, past participle	taken
    VBP	verb be, sing. present, non-3d	am, are
        VVP	verb, sing. present, non-3d	take
    VBZ	verb be, 3rd person sing. present	is
        VVZ	verb, 3rd person sing. present	takes
    VH	verb have, base form	have
    VHD	verb have, past tense	had
    VHG	verb have, gerund/present participle	having
    VHN	verb have, past participle	had
    VHP	verb have, sing. present, non-3d	have
    VHZ	verb have, 3rd person sing. present	has

• Might be differnt from StanfordPosTagger:
    
• Think about how to change "does n't do sth." to "does sth." (i.e. how to change it back)
"""
path_pos = "/playpen/home/tongn/stanford-postagger-full-2017-06-09/"
tagger_pos = "models/english-bidirectional-distsim.tagger"
jar_pos = "stanford-postagger.jar"
st_pos = StanfordPOSTagger(path_pos + tagger_pos, path_pos + jar_pos)

# path_to_jar = 'path_to/stanford-parser-full-2014-08-27/stanford-parser.jar'
# path_to_models_jar = 'path_to/stanford-parser-full-2014-08-27/stanford-parser-3.4.1-models.jar'
# dependency_parser = StanfordDependencyParser(path_to_jar=path_to_jar, path_to_models_jar=path_to_models_jar)

# result = dependency_parser.raw_parse('I shot an elephant in my sleep')
# dep = result.__next__()
# list(dep.triples())

not_token = "not"
def add_not(turn):
    """strategy 6"""    
    utterances = split_lst(turn[:(-1)], eou) # get rid of __eot__

    if utterances == []:
        return turn
    
    adv_utterances = []
    for utterance in utterances:
        if utterance[:(-1)] == []: # if only __eou__ is present
            adv_utterances.append(utterance)
            continue
        
        str_utterance = [index2token[index] for index in utterance] # Convert to text
        
        tagged_utterance = pos_tag(str_utterance[:(-1)]) # get ride of __eou__

        for (token, tag) in tagged_utterance:
            if tag in ["VB", "VBD", "VBG", "VBN", "VBP", "VBZ"]:
                try:
                    i = str_utterance.index(token)
                except:
                    continue

                if tag in ["VB", "VBG", "VBN"]:
                    str_utterance.insert(i, not_token)
                elif tag == "VBD":
                    if token in ["was", "were"]:
#                         str_utterance.insert(i + 1, not_token)
                        str_utterance[i] += "n't"
                    else:
                        present = en.conjugate(token, person=1, tense=en.PRESENT)
                        if present != token and present in vocab:
#                             str_utterance[i: (i + 1)] = ["did", not_token, present]
                            str_utterance[i: (i + 1)] = ["didn't", present]
                        else:
                            continue
                elif tag == "VBP":
#                     if token in ["am", "are"]:
                    if token == "am":
                        str_utterance.insert(i + 1, not_token)
                    elif token == "are":
                        str_utterance[i] += "n't"
                    else:
                        if token in vocab:
#                             str_utterance[i: (i + 1)] = ["do", not_token, token]
                            str_utterance[i: (i + 1)] = ["don't", token]
                        else:
                            continue
                elif tag == "VBZ":
                    if token == "is":
#                         str_utterance.insert(i + 1, not_token)
                        str_utterance[i] += "n't"
                    else:
                        present = en.conjugate(token, person=1, tense=en.PRESENT)
                        if present != token and present in vocab:
#                             str_utterance[i: (i + 1)] = ["does", not_token, present]
                            str_utterance[i: (i + 1)] = ["doesn't", present]
                        else:
                            continue
                break # only need to replace one token for each utterance

        # Convert back to token
        indexed_utterance = [token2index[str_token] for str_token in str_utterance]
        adv_utterances.append(indexed_utterance)
    
    final_turn = list(itertools.chain(*adv_utterances)) + [eot] # append back __eot__

    return final_turn


# In[24]:


# utterance = "i was big . __eou__ i went to class . __eou__ i see some cats . __eou__ he hates you . __eou__ __eot__".lower().split()
# indexed_utterance = [token2index[token] for token in utterance]
# not_utterance = add_not(indexed_utterance)
# print(not_utterance)
# print(" ".join([index2token[index] for index in not_utterance]))


# In[8]:


# def tag_everything(data):
#     dialogues = [
#         split_lst(dialogue, delimiter) 
#         for dialogue in data]


# In[9]:


# """
# Load pickled lists
# """
# data_path = "/playpen/home/tongn/processed_ubuntu/"
# tagged_test_file = data_path + "tagged_test.pkl"
# if Path(tagged_test_file).is_file():
#     tagged_test = load_pickle(tagged_test_file)
# else:
#     test = load_pickle(data_path + "Test.dialogues.pkl")
#     tagged_test = tag_everything(test)


# In[10]:


grammar_error_dict = load_pickle("/playpen/home/tongn/adversarial/data/aesw-2016/bpe_grammar_error_dict.pkl")

def grammar_errors(utterance, adv_rate=1.0):    
    """
    strategy 8:
        Introduce grammatical errors
    """
    utterance = copy.copy(utterance)
    for i in range(len(utterance)):
        if random.random() <= adv_rate:
            try:
                utterance[i] = random.sample(grammar_error_dict[utterance[i]], 1)[0]
            except:
                continue
    return utterance


# In[39]:


pos_tag_dict = {
    'VERB': wordnet.VERB, 
    'NOUN': wordnet.NOUN,
    'ADJ': wordnet.ADJ,
    'ADV': wordnet.ADV}

def antonym(turn):
    """
    strategy 9:
        Replace a word from each utterance with its antonym
    """
    utterances = split_lst(turn[:(-1)], eou) # get rid of __eot__

    if utterances == []:
        return turn
    
    adv_utterances = []
    for utterance in utterances:
        if utterance[:(-1)] == []: # if only __eou__ is present
            adv_utterances.append(utterance)
            continue
        
        str_utterance = [index2token[index] for index in utterance] # Convert to text
        
        tagged_utterance = pos_tag(str_utterance[:(-1)], tagset='universal') # get ride of __eou__

        for (i, (token, tag)) in enumerate(tagged_utterance):
            if token in ['be', 'am', 'is', 'are', 'was', 'were', 'been']:
                continue
            
            try:
                pos = pos_tag_dict[tag]
            except:
                pos = None

            antonyms = [
                lemma.antonyms()[0].name() 
                for synset in wordnet.synsets(token, pos=pos)
                for lemma in synset.lemmas()
                if lemma.antonyms() != []]
            in_vocab_antonyms = set(antonyms).intersection(set(vocab))
            if len(in_vocab_antonyms) == 0: # this includes the case where there are no antonyms
                continue
            else:
                str_utterance[i] = random.sample(in_vocab_antonyms, 1)[0]

            break # only need to replace one token for each utterance

        # Convert back to token
        indexed_utterance = [token2index[str_token] for str_token in str_utterance]
        adv_utterances.append(indexed_utterance)
    
    final_turn = list(itertools.chain(*adv_utterances)) + [eot] # append back __eot__

    return final_turn


# In[41]:


# sent = [token2index[token] for token in "i hate you . __eou__ he was beautiful . __eou__ he left without a word . __eou__ __eot__".split()]
# print(sent)
# print([index2token[index] for index in antonym(sent)])


# In[17]:


# unigram2entity = load_pickle("/home/tongn/Ubuntu-Multiresolution-Tools/ActEntRepresentation/unigram2entity_dict.pkl")
# activities_dict = load_pickle("/home/tongn/Ubuntu-Multiresolution-Tools/ActEntRepresentation/activities_dict.pkl")

# entities = set(unigram2entity.keys()).intersection(set(vocab))
# activities = set(activities_dict.keys()).intersection(set(vocab))

# def confusing_actent(utterance):
#     """
#     strategy 7:
#         Replace entity with a confusing one
#     """
#     utterance_str = [index2token[index] for index in utterance]
#     for (i, token) in enumerate(utterance_str):
#         if token in entities:
#             utterance_str[i] = random.sample(entities, 1)[0]
#             break
#         elif token in activities:
#             utterance_str[i] = random.sample(activities, 1)[0]
#             break
#     utterance = [token2index[token] for token in utterance_str]
    
#     return utterance  


# In[18]:


# utterance = "i was big . __eou__ i went to class . __eou__ i like talking . __eou__ he install hates you . __eou__ __eot__".lower().split()
# indexed_utterance = [token2index[token] for token in utterance]
# not_utterance = confusing_actent(indexed_utterance)
# print(not_utterance)
# print(" ".join([index2token[index] for index in not_utterance]))


# In[14]:


# x = list(grammar_error_dict.items())
# print(len(x))
# print(x[:20])


# In[19]:


"""
Adversarial strategies
"""
no_swap_lst = []

for token in list(string.punctuation) + [eou_token, eot_token]:
    try:
        no_swap_lst.append(token2index[token])
    except:
        pass

# num_stopwords = 100

eou = token2index["__eou__"]
# word_freq_list = read_lines("/home/tongn/vocab_book/word_freq_list.txt")
# stopwords_str = unzip_lst([line.split("\t") for line in word_freq_list])[1][:num_stopwords]
stopwords_str = stopwords.words("english")

# stopwords_str[1:2] = ["is", "are", "am", "was", "were", "been"]
# keep_lst = ["I", "you", "he", "we", "his", "not", "n't", "she", 
#             "what", "their", "who", "her", "my", "when", "which", 
#             "them", "me", "him", "your", "how", "our", "because"]
keep_lst = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", 
            "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 
            'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 
            'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 
            'what', 'which', 'who', 'whom', 'when', 'where', 'why', 'how',
            "not"]

stopwords_str = list(set(stopwords_str).difference(set(keep_lst)))

indexed_stopwords = []
for token in (stopwords_str + list(string.punctuation)):
    if "n't" in token:
        continue
    try:
        indexed_stopwords.append(token2index[token])
    except:
        pass

def get_boundary_indices(utterance):
    boundary_indices = [0] + [i for (i, token) in enumerate(utterance) if token in no_swap_lst] + [len(utterance) - 1]
    return boundary_indices
    
def identity(utterance):
    """
    Identity strategy:
        i.e., do nothing
    """
    return utterance

def random_swap(utterance, swap_ratio=3):
    """
    strategy 1:
        Only swapping neighboring words for each sentence
        New:
            obtain a list of boundary indices, determined by punctuation/__eou__
            between each two boundary indices, randomly swap two tokens
    """
    
    # Avoid modifying the original list
    utterance = copy.deepcopy(utterance)
    
    boundary_indices = get_boundary_indices(utterance)
    
    for (start, end) in zip(boundary_indices[:(-1)], boundary_indices[1:]):
        candidate_indices = set(range(start + 1, end - 1)) # exclude start. end is excluded by default.
#         if len(candidate_indices) != 0:
#             first_letter_of_first_token = utterance[start + 1][0]
#             if not (first_letter_of_first_token >= 'A' and first_letter_of_first_token <= 'Z'): # if the first letter is not capitalized
#                 candidate_indices.add(start + 1)
        num_swaps = (end - start - 1) // swap_ratio
        for i in range(num_swaps):
            if len(candidate_indices) == 0:
                break
            index = random.sample(candidate_indices, 1)[0]
            (utterance[index], utterance[index + 1]) = (utterance[index + 1], utterance[index])
            candidate_indices.remove(index)
            try:
                candidate_indices.remove(index - 1)
            except:
                pass
            try:
                candidate_indices.remove(index + 1)
            except:
                pass

    return utterance

def stopword_dropout(utterance, adv_rate=1.0):
    """strategy 2"""
    utterance = copy.deepcopy(utterance)
    utterance_len = len(utterance)
    
    utterance = [token for token in utterance if token not in indexed_stopwords]
#     max_change = 8
#     count = 0
#     for i in reversed(range(utterance_len)):
#         if utterance[i] in indexed_stopwords:
#             if random.random() <= adv_rate: # randomly drop each word with dropout rate
#                 utterance.pop(i)
#                 count += 1
#                 if count >= max_change:
#                     break
    return utterance

# paraphrase_dict: tuple --> [tuples]
paraphrase_dict = load_pickle(
    "/playpen/home/tongn/ppdb2.0/ppdb-2.0-s-all_paraphrase_dict.pkl")

keys = list(paraphrase_dict.values())
lens = [len(key) for key in keys]
# print(FreqDist(lens).most_common())

def paraphrase(utterance, adv_rate=1.0):
    """strategy 3"""
    utterance = copy.deepcopy(utterance)
    utterance_len = len(utterance)
    
    paraphrased_indices = set() # keep track of which indices are already paraphrased
    for i in range(utterance_len - 1, -1, -1): # reverse order loop
        for j in range(utterance_len, i, -1): # match the longest segment first
            if len(set(range(i, j)).intersection(paraphrased_indices)) == 0: # if no sub-segment of this segment hasn't been paraphrased
                segment = tuple(utterance[i: j])
                try:
                    paraphrased_segment = list(random.choice(paraphrase_dict[segment]))
                    if random.random() <= adv_rate:
                        utterance[i: j] = paraphrased_segment
                        paraphrased_indices.update(list(range(i, j))) # update paraphrased indices
                except:
                    continue
    return utterance

def random_input(utterance):
    """strategy 4"""
    utterance_len = len(utterance)
    return random.sample(indices, utterance_len)

def stammer(utterance, stammer_ratio=3):
    """strategy 5"""
    utterance = copy.deepcopy(utterance)
    utterance_len = len(utterance)
    num_stammers = utterance_len // stammer_ratio
    
    indices = [
        index for index 
        in sorted(random.sample(range(utterance_len), num_stammers), reverse=True)
        if index not in [eou, eot]]

    for index in indices: # from largest to smallest
        utterance[index: (index + 1)] = [utterance[index]] * 2

    return utterance
    
strategies = [
    identity, #0
    random_swap, #1 
    stopword_dropout, #2
    paraphrase, #3
    random_input, #4
    stammer, #5
#     stopword_insertion, #5
    add_not, #6
    generative_paraphrase, #7
#     confusing_actent, #7
    grammar_errors, #8
    antonym, #9
]


# In[3]:


def concat(turns):
    return list(itertools.chain(*turns))

def split_lst_by_length(lst, step_size):
    split = [lst[i: (i + step_size)] for i in range(0, len(lst), step_size)]
    return split


# In[1]:


class DataGenerator(object):
    def __init__(self, data, delimiter, glimpse_training=True, step_size=10, 
                 max_dialogue_length=3, 
                 batch_size=192, shuffle=True, 
                 is_training=True, strategy_index=0,
                 feed_both_examples=False,
                 concat_turns=False,
                 bpe=False):
        self.is_training = is_training
        self.glimpse_training = glimpse_training
        # If we are training and use the truncated version, 
        # we divide data by even chuncks of step_size, otherwise by turns ("__eot__")
        if bpe:
            self.dialogues = data
            strategy_index = 0 # no need to apply any strategy
        else:
            print("Dividing each dialogue by turns (i.e. __eot__)")
            self.dialogues = [
                split_lst(dialogue, delimiter)
                for dialogue in data]

        self.max_dialogue_length = max_dialogue_length
                    
        self.shuffle = shuffle
        if self.shuffle:
            random.shuffle(self.dialogues)
        
        self.strategy_index = strategy_index
        
        self.feed_both_examples = feed_both_examples
        if self.feed_both_examples:
            assert strategy_index != 0
            assert batch_size % 2 == 0
            self.batch_size = batch_size // 2 # because we concatenate pos + neg examples in each batch
        else:
            self.batch_size = batch_size
            
        self.concat_turns = concat_turns

    @staticmethod
    def max_len(lst):
        return max([len(x) for x in lst])
    
    @staticmethod
    def to_str(batch_examples):
        str_batch_examples = [
            [" ".join([str(token) for token in turn]) 
             for turn in example]
            for example in batch_examples]
        return str_batch_examples
    
    def batch_generator(self):       
        print("There are %d dialogues." % len(self.dialogues))
        
        turn_lengths_lst = [
            len(turn) 
            for dialogue in self.dialogues
            for turn in dialogue]
        
        if not self.is_training:
            print("We are testing ==> no length threshold is applied.")
        
        length_threshold = int(np.percentile(turn_lengths_lst, 90)) if self.is_training else max(turn_lengths_lst)
        print("Length threshold:", length_threshold)
        print("All turns longer than this will be truncated to this length.")
        
        # Convert all indices to string_indices
        self.dialogues = [
            [[token
              for token in turn[(-length_threshold):]] # only keeping the last length_threshold tokens
             for turn in dialogue] 
            for dialogue in self.dialogues
            if len(dialogue) >= self.max_dialogue_length]
        
        num_dialogues = len(self.dialogues)
        print("There are %d dialogues left." % num_dialogues)
        
        assert num_dialogues >= self.batch_size, "Error: Number of dialogues less than batch_size"
    
        acc = self.dialogues[:self.batch_size] # an accumlator that yields a batch when it is of batch_size
        dialogue_indices = list(range(self.batch_size)) # initialize dialogue indices
        next_dialogue_index = self.batch_size # initialize the index of the next dialogue
        start_turn_indices = [0] * self.batch_size # track which turn we are at
        while True:
            batch_examples = [
                dialogue[start_turn_index: (start_turn_index + self.max_dialogue_length)] 
                for (dialogue, start_turn_index) 
                in zip(acc, start_turn_indices)]
            
            if self.strategy_index == 4: # if we are using random_source_input
                adv_batch_examples = [
                    [random.choice(random.choice(self.dialogues))
                     if (i != self.max_dialogue_length - 1) 
                     else turn # avoid modifying target turn
                     for (i, turn) in enumerate(example)]
                    for example in batch_examples]
            elif self.strategy_index == 5: # if we are using random ground-truth target
                adv_batch_examples = [
                    [turn
                     if (i != self.max_dialogue_length - 1) 
                     else random.choice(random.choice(self.dialogues)) # modify target turn
                     for (i, turn) in enumerate(example)]
                    for example in batch_examples]
            else:
                adv_batch_examples = [
                    [strategies[self.strategy_index](turn)
                     if (i != self.max_dialogue_length - 1) 
                     else turn # avoid modifying target turn
                     for (i, turn) in enumerate(example)]
                    for example in batch_examples]
            
            if self.feed_both_examples:
                feed_dialogue_indices = dialogue_indices + dialogue_indices
                feed_start_turn_indices = start_turn_indices + start_turn_indices
                feed_batch_examples = batch_examples + adv_batch_examples
            else:
                feed_dialogue_indices = dialogue_indices
                feed_start_turn_indices = start_turn_indices
                feed_batch_examples = adv_batch_examples
                    
            turn_lengths_lst = [
                [len(turn) for turn in example]
                for example in feed_batch_examples]
            
            if self.concat_turns:
                turn_lengths_lst = [
                    [sum(turn_lengths[:(-1)]), turn_lengths[-1]]
                    for turn_lengths in turn_lengths_lst]
            
            # When cancating turns, yield [int] rather than str
            if self.concat_turns:
                feed_batch_examples = [
                    [concat(example[:(-1)]), example[-1]]
                    for example in feed_batch_examples]
            else:
                feed_batch_examples = DataGenerator.to_str(feed_batch_examples)
            
            yield (feed_dialogue_indices,
                   feed_start_turn_indices, 
                   feed_batch_examples, 
                   turn_lengths_lst)
            
            for (i, dialogue) in enumerate(acc):
                start_turn_indices[i] += 1 # move on to the next example
                if start_turn_indices[i] + self.max_dialogue_length > len(dialogue): # if we finish the current dialogue
                    acc[i] = self.dialogues[next_dialogue_index] # change examples at index i in acc
                    dialogue_indices[i] = next_dialogue_index
                    start_turn_indices[i] = 0 # reset example index
                    next_dialogue_index += 1
                    if next_dialogue_index >= num_dialogues:
                        """todo: let the remaining dialgoues finish when out of new dialgoues"""
                        yield None
                        return
        
                            
#             # Apply adversarial strategy
#             examples = [
#                 (strategies[self.strategy_index](turn1),
#                  strategies[self.strategy_index](turn2),
#                  turn3)
#                 for (turn1, turn2, turn3) in examples] 
                            
#             if concat_turns:
#                 examples = [
#                     (turn1 + turn2, turn3) 
#                     for (turn1, turn2, turn3)
#                     in examples
#                     if len(turn3) <= self.max_iterations]
            
#             num_examples = len(examples)
#             acc_size = len(acc)
#             if acc_size + num_examples < self.batch_size:
#                 acc += examples
#                 continue
#             else:
#                 concat_lst = acc + examples
#                 acc = concat_lst[self.batch_size:]
#                 yield concat_lst[:self.batch_size


# In[4]:


# from src.basic.util import load_pickles, unzip_lst

# def get_vocab(data):
#     vocab = unzip_lst(data)[0]
#     return vocab

# """
# Load pickled lists
# """
# data_path = "/playpen/home/tongn/processed_ubuntu/"

# filenames = [
#     "Dataset.dict.pkl", 
#     "Training.dialogues.pkl",
#     "Validation.dialogues.pkl",
#     "Test.dialogues.pkl"
# ]

# files = [data_path + filename for filename in filenames]

# # Load files
# data = load_pickles(files)

# vocab = get_vocab(data[0])

# splits = ["train", "valid", "test"]
# data_dict = {
#     split: dialogues for (split, dialogues) 
#     in zip(splits, data[1:])}

# end_token_str = "__eot__"
# end_token = vocab.index(end_token_str)


# In[5]:


# generator = DataGenerator(
#     data_dict["train"], end_token,
#     20001, step_size=20,
#     max_dialogue_length=3, max_iterations=None, 
#     batch_size=8, shuffle=True, 
#     is_training=True,
#     strategy_index=0)
# batch = generator.batch_generator()


# In[6]:


# pprint(next(batch)[1])

