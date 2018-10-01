
# coding: utf-8

# In[5]:


# Imports for compatibility between Python 2&3
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from six.moves import xrange
import numpy as np
import tensorflow as tf
import os
from os.path import join
import sys
from pprint import pprint
from pathlib import Path
from math import exp
import random
import itertools
import string
from pprint import pprint
import argparse
sys.path.extend([".", ".."])
from src.basic.ubuntu_data_generator import DataGenerator
from src.model.util import gpu_config 
from src.basic.util import (shuffle, remove_duplicates, pad,
                            unzip_lst, zip_lsts,
                            prepend, append,
                            load_pickle, load_pickles, 
                            dump_pickle, dump_pickles, 
                            build_dict, read_lines, write_lines, 
                            group_lst, decode2string)
from src.model.vhred import VHRED


# In[ ]:


should_not_change = [
    "random_swap", 
    "stopword_dropout", 
    "paraphrase", 
    "generative_paraphrase",
    "grammar_errors", 
]
should_change = [
    "add_not",
    "antonym"
]

strategies = ["normal_inputs"] + should_not_change + should_change


# In[ ]:


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train/Val/Test various models on the Ubuntu dataset")
    parser.add_argument(
        "--start_epoch", type=int, default=-1, 
        help="which epoch to restore from (-1 means starting from scratch)")
    parser.add_argument(
        "--num_epochs", type=int, default=80, 
        help="number of epochs to train")
    parser.add_argument(
        "--infer_only", action="store_true",
        help="if we are only inferring on the test set")
    parser.add_argument(
        "--gpu_start_index", help="which gpu(s) to use",
        type=int, default=0)
    parser.add_argument(
        "--train_strategy", type=str, 
        choices=strategies, default=strategies[0],
        help="strategy used when training")
    parser.add_argument(
        "--test_strategy", type=str, 
        choices=strategies, default=strategies[0],
        help="strategy used when testing")
    parser.add_argument(
        "--batch_size", help="batch size", 
        type=int, default=64)
    parser.add_argument(
        "--force_restore_point", type=str, default="",
        help="a point that we restore from")
    parser.add_argument(
        "--force_store_point", type=str, default="",
        help="a point that we store the model")
    parser.add_argument(
        "--max_margin_weight", type=float, default=1.0,
        help="weight of max margin")
    parser.add_argument(
        "--margin", type=float, default=0.1,
        help="margin as in max margin training")
    parser.add_argument(
        "--bpe", action="store_true",
        help="if we use bpe encoding")

    args = parser.parse_args()
    return args


# In[ ]:


ckpt_path = "ckpt"
output_path = "output"
data_path = "data"

args = parse_args()
start_epoch = args.start_epoch
num_epochs = args.num_epochs
infer_only = args.infer_only
gpu_start_index = args.gpu_start_index
train_strategy = args.train_strategy
test_strategy = args.test_strategy
batch_size = args.batch_size
force_restore_point = args.force_restore_point
force_store_point = args.force_store_point
max_margin_weight = args.max_margin_weight
margin = args.margin
bpe = args.bpe

if bpe:
    print("Using bpe-processed data...")

assert num_epochs >= 1, "num_epochs cannot be less than 1"

if infer_only:
    print("infer_only option on...")


# In[ ]:


strategy = test_strategy if infer_only else train_strategy
if infer_only:
    use_max_margin = False
    feed_both_examples = False
else:
    use_max_margin = (strategy in should_change)
    feed_both_examples = use_max_margin

if use_max_margin:
    print("This is a Should-Change strategy, adding in max margin objective.")

if feed_both_examples:
    print("Feeding both normal and adversarial examples")
    
max_dialogue_length = 3

"""
gpu configurations
"""
num_gpus = 1
assert batch_size % num_gpus == 0

"""
flags
"""
no_unk = True

# Extra strings to prepend to file name
bpe_str = "bpe_" if bpe else ""
model_extra_str = f"{bpe_str}vhred_train_{train_strategy}"
output_extra_str = f"{model_extra_str}_test_{test_strategy}"
print("Model extra string:", model_extra_str)
print("Output extra string:", output_extra_str)


# In[ ]:


def get_vocab(bpe_str=""):
    filename = f"{bpe_str}vocab.txt"
    file = os.path.join(data_path, filename)
    vocab = read_lines(file)
    return vocab


# In[ ]:


def get_data_dict(strategy_str, bpe_str=""):
    filenames = [
        f"indexed/{bpe_str}indexed_training_text.{strategy_str}.pkl",
        f"indexed/{bpe_str}indexed_valid_text.{strategy_str}.pkl",
        f"indexed/{bpe_str}indexed_test_text.{strategy_str}.pkl"]
    files = [os.path.join(data_path, filename) for filename in filenames]
    data = load_pickles(files)
    data_dict = {
        split: dialogues for (split, dialogues) 
        in zip(splits, data[:3])}
    return data_dict


# In[ ]:


"""
Load pickled lists
"""
splits = ["train", "valid", "test"]
strategy_str = test_strategy if infer_only else train_strategy
vocab = get_vocab(bpe_str=bpe_str)
norm_data_dict = get_data_dict(strategies[0], bpe_str=bpe_str)
adv_data_dict = get_data_dict(strategy_str, bpe_str=bpe_str)


# In[ ]:


start_token_str = "__sot__"
vocab.append(start_token_str)
eou_str = "__eou__"
end_token_str = "__eot__"
unk_token_str = "**unknown**"

vocab_size = len(vocab)

print("Total vocab size:", vocab_size)

start_token = vocab.index(start_token_str)
end_token = vocab.index(end_token_str)
unk_token = vocab.index(unk_token_str)
unk_indices = [unk_token]

# Index vocabulary
index2token = {i: token for (i, token) in enumerate(vocab)}
token2index = {token: i for (i, token) in enumerate(vocab)}


# In[ ]:


"""
hyperparameters
"""
beam_width = 10
clipping_threshold = 2.0 # threshold for gradient clipping
# clipping_threshold = 5.0 # threshold for gradient clipping
embedding_size = 300
learning_rate = 0.0002
hidden_size_encoder = 256
hidden_size_context = 512
hidden_size_decoder = 512
num_layers_encoder = 1
num_layers_context = 1
num_layers_decoder = 1
"""think about max_iterations: will it hurt hred in some way?"""
# max_iterations = 44 if bpe else 42 # should be computed by 95 percentile of all sequence lengths
max_iterations = 48 if bpe else 42 # should be computed by 95 percentile of all sequence lengths
dropout_rate = 0.2
attention_size = 512
attention_layer_size = 256


# In[ ]:


def build_model():
    tf.reset_default_graph()
    graph = tf.Graph()
    with graph.as_default():
        model = VHRED(
            batch_size,
            vocab_size,
            embedding_size,
            hidden_size_encoder, hidden_size_context, hidden_size_decoder,
            max_iterations, max_dialogue_length,
            start_token, end_token, unk_indices,
            num_layers_encoder=1, num_layers_context=1, num_layers_decoder=1,
            attention_size=attention_size, attention_layer_size=attention_layer_size,
            gpu_start_index=gpu_start_index, 
            learning_rate=learning_rate, 
            clipping_threshold=clipping_threshold,
            feed_both_examples=feed_both_examples,
            use_max_margin=use_max_margin, max_margin_weight=max_margin_weight, margin=margin
        )
        saver_seq2seq = tf.train.Saver(var_list=model.trainable_variables + [model.global_step])
    print("Done building model graph.")
    return (model, graph, saver_seq2seq)


# In[ ]:


def concat(turns):
    turns_concat = turns[0]
    for turn in turns[1:]:
        turns_concat += (' ' + turn)
    return turns_concat

def get_source_and_target(examples):
    source = []
    target = []
    for example in examples:
        source_str = concat(example[:(-1)])
        target_str = example[-1]
        source.append(
            list(map(int, source_str.split())))
        target.append(
            list(map(int, target_str.split())))
    return (source, target)


# In[ ]:


def run_vhred(model, sess, mode, epoch, saver=None):
    training_flag = True if mode == "train" else False
    norm_dialogues = norm_data_dict[mode]
    adv_dialogues = adv_data_dict[mode]
    
    generator = DataGenerator(
        norm_dialogues,
        adv_dialogues=adv_dialogues,
        feed_both_examples=feed_both_examples,
        is_training = training_flag,
        batch_size=batch_size,
        max_dialogue_length=max_dialogue_length)
    batch = generator.batch_generator()
    print("Initialized data generator.")
    
    responses = []
    total_loss = 0.0
    adv_total_loss = 0.0
    total_num_tokens = 0.0
    batch_counter = 0    
    if mode != "train":
        source_lst = []
        target_lst = []
        dialogue_indices_lst = []
        start_turn_indices_lst = []
    
    if use_max_margin:
        avg_margin = 0.0
        
    while True:
        next_batch = next(batch)
        if next_batch is None:
            break
            
        # if it's os, we always set start_turn_indices to 0
        (dialogue_indices, start_turn_indices, examples, turn_lengths_lst) = next_batch
                
        feed_dict_seqs = {
            model.dialogue: examples,
            model.turn_length: turn_lengths_lst,
            model.start_turn_index: start_turn_indices,
            model.start_tokens: [start_token] * batch_size
        }
                
        if mode == "train":
            if use_max_margin:
                fetches = [model.batch_total_loss, 
                           model.batch_num_tokens,
                           model.apply_gradients_op, 
                           model.avg_margin_loss, # testing
                           model.global_step]
            else:
                fetches = [model.batch_total_loss, 
                           model.batch_num_tokens,
                           model.apply_gradients_op, 
                           model.global_step]
            feed_dict = {
                model.keep_prob: 1 - dropout_rate,
                model.is_training: training_flag}
            
            result = sess.run(
                fetches, 
                feed_dict={**feed_dict_seqs, **feed_dict})
            
            if use_max_margin:
                avg_margin = (avg_margin * batch_counter + result[-2]) / (batch_counter + 1)  
                print("Avg margin (this should be getting smaller (or getting larger in abs. value) over time):", 
                      avg_margin)
    
            if feed_both_examples:
                (loss, adv_loss) = result[0]
            else:
                loss = result[0]
            
            average_log_perplexity = loss / result[1]
            total_loss += loss
            total_num_tokens += result[1]
            print("Epoch (%s) %d, Batch %d, Global step %d:" % 
                  (mode, epoch, batch_counter, result[-1]))
            print("Perplexity: %.2f" % exp(average_log_perplexity))
            print("Perplexity so far:", exp(total_loss / total_num_tokens))
            
            if feed_both_examples:
                adv_average_log_perplexity = adv_loss / result[1]
                adv_total_loss += adv_loss
                print("Adv-perplexity: %.2f" % exp(adv_average_log_perplexity))
                print("Adv-perplexity so far:", exp(adv_total_loss / total_num_tokens))
        else:
            (source, target) = get_source_and_target(examples)          
            source_lst.extend(source) 
            target_lst.extend(target)
            
            dialogue_indices_lst.extend(dialogue_indices)
            start_turn_indices_lst.extend(start_turn_indices)
            
            feed_dict = {
                model.keep_prob: 1.0,
                model.is_training: training_flag}
            (ids, lengths) = sess.run(
                [model.batch_sample_ids_beam, model.batch_final_lengths_beam],
                feed_dict={**feed_dict_seqs, **feed_dict})

            batch_responses = [
                [index for index in response[:length]]
                for (response, length) 
                in zip(ids.tolist(), lengths.tolist())]
            responses.extend(batch_responses)
            print("Finished testing batch %d" % batch_counter)
        
        batch_counter += 1
            
    if mode == "train":
        epoch_perplexity = total_loss / total_num_tokens
        print("Epoch (%s) %d average perplexity: %.2f" % 
              (mode, epoch, exp(epoch_perplexity)))
        
        if force_store_point == "":
            store_ckpt = os.path.join(ckpt_path, f"{model_extra_str}_{epoch}")
        else:
            store_ckpt = force_store_point
        saver.save(sess, store_ckpt)
        print(f"Checkpoint saved for epoch {epoch}.")
    else:
        zipped = zip_lsts(
            [dialogue_indices_lst, start_turn_indices_lst, 
             source_lst, target_lst, responses])
        zipped.sort(key=lambda x: x[:2]) # sort on dialogue indices & start_turn_indices
        zipped_responses = zip_lsts(unzip_lst(zipped)[2:])
        return zipped_responses


# In[ ]:


def maybe_add(sent, token):
    if sent.strip() == "":
        return eou_str
    elif sent[-len(eou_str):] != eou_str:
        return sent + " " + eou_str
    else:
        return sent


# In[ ]:


def main(start_epoch):
    (model, graph, saver_seq2seq) = build_model()
    config = gpu_config()

    with tf.Session(graph=graph, config=config) as sess:
        sess.run(tf.global_variables_initializer())
        print("Initialized.")

        restore_ckpt = None
        if start_epoch > -1:
            if force_restore_point != "":
                restore_ckpt = force_restore_point
            else:
                restore_ckpt = f"{ckpt_path}/{model_extra_str}_{start_epoch}"

        if restore_ckpt is not None:
            saver_seq2seq.restore(sess, restore_ckpt)
            print("Restored from", restore_ckpt)

        for i in xrange(num_epochs):
            if not infer_only: # for getting perplexity of test data, use train branch
                mode = "train"
                start_epoch += 1
                run_vhred(model, sess, mode, start_epoch, saver=saver_seq2seq)

                if not no_validation and not glimpse_training and start_epoch % 5 == 0 and start_epoch >= 10:
                    mode = "valid"
                    zipped_responses = run_vhred(model, sess, mode, start_epoch)
                else:
                    continue
            else:
                print("Inferring on test set...")
                mode = "test"
                zipped_responses = run_vhred(model, sess, mode, start_epoch)

            # Make sure sent is not empty and always ends with an eou
            flattened = [decode2string(index2token, sent, end_token=end_token_str, remove_END_TOKEN=True) 
                         for tp in zipped_responses for sent in tp]
            flattened = [maybe_add(sent, eou_str) for sent in flattened]

            # now we mark sentences that are generated by our model
            marked_G = [("G: " + sent) 
                        if k % 3 == 1 else sent
                        for (k, sent) in enumerate(flattened)]

            marked_M = [("M: " + sent) 
                        if k % 3 == 2 else sent
                        for (k, sent) in enumerate(marked_G)]

            file = os.path.join(output_path, f"{output_extra_str}_{mode}_result_{start_epoch}.txt")

            write_lines(file, marked_M)

            # only need 1 epoch for inferring or getting PPL
            if infer_only: 
                break


# In[ ]:


if __name__ == "__main__":
    main(start_epoch)

