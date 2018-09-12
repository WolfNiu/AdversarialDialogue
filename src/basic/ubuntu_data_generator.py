
# coding: utf-8

# In[1]:


"""
Todo: combine read_lines, load_pickle, etc... to one single function load_file(),
    and use if statement to see which suffix the file has. Also keep an optional param
    suffix=None just in case we want to force it to load with a certain format

"""
from random import shuffle
import numpy as np


# In[2]:


def to_str(batch_examples):
    str_batch_examples = [
        [" ".join([str(token) for token in turn]) 
         for turn in example]
        for example in batch_examples]
    return str_batch_examples


# In[3]:


class DataGenerator(object):
    def __init__(self,
                 norm_dialogues,
                 adv_dialogues=None,
                 feed_both_examples=False,
                 is_training=True, 
                 batch_size=192,
                 max_dialogue_length=3):
        self.norm_dialogues = norm_dialogues
        self.adv_dialogues = adv_dialogues
        self.feed_both_examples = feed_both_examples
        self.is_training = is_training
            
        if self.feed_both_examples: # if we are not feeding both examples
            assert batch_size % 2 == 0
            assert norm_dilogues is not None, "Error: feeding both examples, need norm dialogues too."
            self.batch_size = batch_size // 2 # because we concatenate pos + neg examples in each batch
        else:
            self.batch_size = batch_size
        
        self.max_dialogue_length = max_dialogue_length
    
    def batch_generator(self):       
        print(f"There are {len(self.norm_dialogues)} dialogues.")
        
        turn_lengths_lst = [
            len(turn) 
            for dialogue in self.norm_dialogues
            for turn in dialogue]
        
        if not self.is_training:
            print("We are testing ==> no length threshold is applied.")
        
        length_threshold = int(np.percentile(turn_lengths_lst, 90)) if self.is_training else max(turn_lengths_lst)
        print("Length threshold:", length_threshold)
        print("All turns longer than this will be truncated to this length.")
        
        # Truncate based on length threshold
        norm_dialogues = [
            [turn[(-length_threshold):] # only keeping the last length_threshold tokens
             for turn in dialogue] 
            for dialogue in self.norm_dialogues
            if len(dialogue) >= self.max_dialogue_length]
        
        if self.norm_dialogues is not None:
            adv_dialogues = [
                [turn[(-length_threshold):] # only keeping the last length_threshold tokens
                 for turn in dialogue] 
                for dialogue in self.adv_dialogues
                if len(dialogue) >= self.max_dialogue_length]
        
        num_dialogues = len(norm_dialogues)
        print(f"There are {num_dialogues} dialogues left.")
        
        assert num_dialogues >= self.batch_size, "Error: Number of dialogues less than batch_size"
        
        if self.is_training: # only shuffle dataset if we are training
            if self.adv_dialogues is None:
                shuffle(norm_dialogues)
            else:
                zipped = list(zip(norm_dialogues, adv_dialogues))
                shuffle(zipped)
                norm_dialogues, adv_dialogues = list(zip(*zipped))
        
        dialogue_indices = list(range(self.batch_size)) # initialize dialogue indices
        next_dialogue_index = self.batch_size # initialize the index of the next dialogue
        start_turn_indices = [0] * self.batch_size # track which turn we are at
        while True:
            norm_batch_examples = [
                norm_dialogues[dialogue_index][start_turn_index: (start_turn_index + self.max_dialogue_length)] 
                for (dialogue_index, start_turn_index) 
                in zip(dialogue_indices, start_turn_indices)]
            
            if self.adv_dialogues is not None:
                # Avoid modifying target turn
                adv_batch_examples = [
                    (adv_dialogues[dialogue_index][start_turn_index: (start_turn_index + self.max_dialogue_length - 1)] 
                     + norm_dialogues[dialogue_index][(start_turn_index + self.max_dialogue_length - 1): (start_turn_index + self.max_dialogue_length)])
                    for (dialogue_index, start_turn_index) 
                    in zip(dialogue_indices, start_turn_indices)]
            
            if self.feed_both_examples:
                feed_dialogue_indices = dialogue_indices + dialogue_indices
                feed_start_turn_indices = start_turn_indices + start_turn_indices
                feed_batch_examples = norm_batch_examples + adv_batch_examples
            else:
                feed_dialogue_indices = dialogue_indices
                feed_start_turn_indices = start_turn_indices
                feed_batch_examples = adv_batch_examples
                    
            turn_lengths_lst = [
                [len(turn) for turn in example]
                for example in feed_batch_examples]
            
            yield (feed_dialogue_indices,
                   feed_start_turn_indices, 
                   to_str(feed_batch_examples), 
                   turn_lengths_lst)
            
            for i in range(self.batch_size):
                start_turn_indices[i] += 1 # move on to the next example
                # If we've finished the current dialogue
                if start_turn_indices[i] + self.max_dialogue_length > len(norm_dialogues[dialogue_indices[i]]):
                    dialogue_indices[i] = next_dialogue_index # move on to the next dialogue
                    start_turn_indices[i] = 0 # reset example index
                    next_dialogue_index += 1
                    if next_dialogue_index >= num_dialogues:
                        """todo: let the remaining dialgoues finish when out of new dialgoues"""
                        yield None
                        return

