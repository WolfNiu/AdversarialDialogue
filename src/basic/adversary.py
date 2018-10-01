
# coding: utf-8

# In[5]:


from nltk import pos_tag
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet, stopwords
from util import split_lst, read_lines, remove_duplicates, load_pickle, dump_pickle
from itertools import chain
from pattern import en
import random
import string
from copy import copy
from tqdm import tqdm
from os.path import isfile


# In[2]:


"""constants"""

eou = "__eou__"
eot = "__eot__"

pos_tag_dict = {
    'VERB': wordnet.VERB, 
    'NOUN': wordnet.NOUN,
    'ADJ': wordnet.ADJ,
    'ADV': wordnet.ADV}

no_swap_lst = list(string.punctuation) + [eou, eot] # for random swap

stopword_keep_lst = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", 
    "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 
    'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 
    'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 
    'what', 'which', 'who', 'whom', 'when', 'where', 'why', 'how',
    "not"]


# In[3]:


class Adversary(object):
    def __init__(self,
                 vocab):
        self.vocab = vocab
        self.strategy_fn_dict = {
            "normal_inputs": self._normal_inputs,
            "random_swap": self._random_swap,
            "stopword_dropout": self._stopword_dropout,
            "add_not": self._add_not,
            "antonym": self._antonym,
            "paraphrase": self._paraphrase,
        } 
        self.stopwords = list(
            set(stopwords.words("english"))
            .difference(set(stopword_keep_lst)))
        self.paraphrase_dict = None
    
    def _get_boundary_indices(self, utterance):
        boundary_indices = (
            [0] + 
            [i for (i, token) in enumerate(utterance) if token in no_swap_lst] + 
            [len(utterance) - 1])
        return boundary_indices
    
    def apply_strategy(self, strategy_name, dialogue):
        adv_turns = []
        for turn in dialogue.split(eot):
            adv_utterances = []
            for utterance in turn.strip().split(eou):
                strategy_fn = self.strategy_fn_dict[strategy_name]
                stripped_utterance = utterance.strip()
                if stripped_utterance == "":
                    adv_utterance = utterance
                    adv_utterances.append(adv_utterance)
                else:
                    adv_utterance = strategy_fn(stripped_utterance.split()) # adv_utterance: str
                    adv_utterances.append(' '.join(adv_utterance))
            adv_turn = (' ' + eou + ' ').join(adv_utterances).strip() # append an extra eou
            adv_turns.append(adv_turn)
        adv_dialogue = (' ' + eot + ' ').join(adv_turns) # no need to append an extra eot
        return adv_dialogue
    
    def _normal_inputs(self, utterance):
        return utterance
    
    def _random_swap(self, utterance, swap_ratio=3):
        """
        strategy 1:
            Only swapping neighboring words for each sentence
            New:
                obtain a list of boundary indices, determined by punctuation or __eou__
                between each two boundary indices, randomly swap two tokens
        """

        utterance = copy(utterance)
        # Avoid modifying the original list
        boundary_indices = self._get_boundary_indices(utterance)

        for (start, end) in zip(boundary_indices[:(-1)], boundary_indices[1:]):
            candidate_indices = set(range(start + 1, end - 1)) # exclude start. end is excluded by default.
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
    
    def _stopword_dropout(self, utterance):
        adv_utterance = [
            token for token in utterance 
            if token not in self.stopwords]
        return adv_utterance
    
    def _contains_unknown(self, tokens):
        result = any(
            [(token not in self.vocab) for token in tokens])
        return result
    
    def _build_para_dict(self):
        path = "data/ppdb-2.0-s-all"
        lines = read_lines(path)
        relations = [line.split(" ||| ")[-1] for line in lines]
        equivalent_pairs = []
        print("Preprocessing raw data...")
        for line in tqdm(lines):
            split = line.split(" ||| ")    
            if split[-1] == "Equivalence":
                equivalent_pairs.append(tuple(split[1:3]))        

        paraphrase_pairs = [line.split(" ||| ")[1:3] for line in lines]
        equivalent_pairs_ubuntu = []
        print("Extracting paraphrase pairs...")
        for pair in tqdm(equivalent_pairs):
            tokens_0 = word_tokenize(pair[0]) 
            tokens_1 = word_tokenize(pair[1])
            if not (self._contains_unknown(tokens_0) or self._contains_unknown(tokens_1)):
                equivalent_pairs_ubuntu.append(
                    (tokens_0, tokens_1))
        
        # Insert paraphrases in both directions
        print("Building dictionary...")
        self.paraphrase_dict = {}
        for (p0, p1) in tqdm(equivalent_pairs_ubuntu):
            p0 = tuple(p0)
            p1 = tuple(p1)
            try:
                self.paraphrase_dict[p0] = self.paraphrase_dict[p0] + [p1]
            except:
                self.paraphrase_dict[p0] = [p1]

            try:
                self.paraphrase_dict[p1] = self.paraphrase_dict[p1] + [p0]
            except:
                self.paraphrase_dict[p1] = [p0]
             
        # Remove duplicated paraphrases
        for key in self.paraphrase_dict:
            self.paraphrase_dict[key] = remove_duplicates(self.paraphrase_dict[key])
    
    def _paraphrase(self, utterance, adv_rate=1.0):
        """strategy 3"""
        if self.paraphrase_dict is None:
            file = "data/paraphrase_dict.pkl"
            if isfile(file):
                self.paraphrase_dict = load_pickle(file)
            else:
                print("Creating paraphrase dictionary from PPDB...")
                self._build_para_dict()
                print(f"Done with paraphrase dictionary size: {len(self.paraphrase_dict)}")
                dump_pickle(file, self.paraphrase_dict)
            
        utterance = copy(utterance)
        utterance_len = len(utterance)

        paraphrased_indices = set() # keep track of which indices are already paraphrased
        for i in range(utterance_len - 1, -1, -1): # reverse order loop
            for j in range(utterance_len, i, -1): # match the longest segment first
                if len(set(range(i, j)).intersection(paraphrased_indices)) == 0: # if no sub-segment of this segment hasn't been paraphrased
                    segment = tuple(utterance[i: j])
                    try:
                        paraphrased_segment = list(random.choice(self.paraphrase_dict[segment]))
                        if random.random() <= adv_rate:
                            utterance[i: j] = paraphrased_segment
                            paraphrased_indices.update(list(range(i, j))) # update paraphrased indices
                    except:
                        continue
        return utterance

    def _add_not(self, utterance):
        """
        strategy 6
        """
        utterance = copy(utterance)
        tagged_utterance = pos_tag(utterance)
        for token, tag in tagged_utterance:            
            if tag in ["VB", "VBD", "VBG", "VBN", "VBP", "VBZ"]:
                try:
                    i = utterance.index(token)
                except:
                    continue

                if tag in ["VB", "VBG", "VBN"]:
                    utterance.insert(i, "not")
                elif tag == "VBD":
                    if token in ["was", "were"]:
#                         utterance.insert(i + 1, "not")
                        utterance[i] += "n't"
                    else:
                        present = en.conjugate(token, person=1, tense=en.PRESENT)
                        if present != token and present in self.vocab:
#                             utterance[i: (i + 1)] = ["did", "not", present]
                            utterance[i: (i + 1)] = ["didn't", present]
                        else:
                            continue
                elif tag == "VBP":
#                     if token in ["am", "are"]:
                    if token == "am":
                        utterance.insert(i + 1, "not")
                    elif token == "are":
                        utterance[i] += "n't"
                    else:
                        if token in self.vocab:
#                             utterance[i: (i + 1)] = ["do", "not", token]
                            utterance[i: (i + 1)] = ["don't", token]
                        else:
                            continue
                elif tag == "VBZ":
                    if token == "is":
#                         utterance.insert(i + 1, "not")
                        utterance[i] += "n't"
                    else:
                        present = en.conjugate(token, person=1, tense=en.PRESENT)
                        if present != token and present in self.vocab:
#                             utterance[i: (i + 1)] = ["does", "not", present]
                            utterance[i: (i + 1)] = ["doesn't", present]
                        else:
                            continue
                break # only need to replace one token for each utterance
        return utterance

    def _antonym(self, utterance):
        """
        strategy 9:
            Replace a word from each utterance with its antonym
        """
        utterance = copy(utterance)
        tagged_utterance = pos_tag(utterance, tagset="universal")
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
            in_vocab_antonyms = set(antonyms).intersection(set(self.vocab))
            if len(in_vocab_antonyms) == 0: # this includes the case where there are no antonyms
                continue
            else:
                utterance[i] = random.sample(in_vocab_antonyms, 1)[0]
                break # only need to replace one token for each utterance

        return ' '.join(utterance)


# In[4]:


# from util import load_pickle
# path = "/Users/mrnt0810/UbuntuDialogueCorpus/"
# dataset_dict_file = "Dataset.dict.pkl"
# dataset_dict = load_pickle(path + dataset_dict_file)
# index2token = {index: token for (token, index, _, _) in dataset_dict}
# token2index = {token: index for (token, index, _, _) in dataset_dict}
# vocab = list(token2index.keys())
# indices = list(token2index.values())


# In[5]:


# adv = Adversary(vocab)
# dialogue = "i think we could import the old comments via rsync , but from there we need to go via email . I think it is easier than caching the status on each bug and than import bits here and there __eou__ __eot__ it would be very easy to keep a hash db of **unknown** __eou__ sounds good __eou__ __eot__ ok __eou__ perhaps we can ship an ad-hoc **unknown** __eou__ __eot__ version ? __eou__ __eot__ thanks __eou__ __eot__ not yet __eou__ it is covered by your insurance ? __eou__ __eot__ yes __eou__ but it 's really not the right time :/ __eou__ with a changing house upcoming in 3 weeks __eou__ __eot__ you will be moving into your house soon ? __eou__ posted a message recently which explains what to do if the autoconfiguration does not do what you expect __eou__ __eot__ how urgent is **unknown** ? __eou__ __eot__ not particularly urgent , but a policy violation __eou__ __eot__ i agree that we should kill the -novtswitch __eou__ __eot__ ok __eou__ __eot__ would you consider a package split a feature ? __eou__ __eot__ context ? __eou__ __eot__ splitting xfonts* out of **unknown** . one upload for the rest of the life and that 's it __eou__ __eot__ splitting the source package you mean ? __eou__ __eot__ yes . same binary packages . __eou__ __eot__ I would prefer to avoid it at this stage . this is something that has gone into **unknown** svn , I assume ? __eou__ __eot__ basically each xfree86 upload will NOT force users to upgrade 100Mb of fonts for nothing __eou__ no something i did in my spare time . __eou__"
# adv_dialogue = adv.apply_strategy("random_swap", dialogue)
# print(dialogue)
# print("")
# print(adv_dialogue)

