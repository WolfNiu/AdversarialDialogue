
# coding: utf-8

# In[1]:


"""
This model differs from traditional MMI in that the backward model predicts previous two turns rather than just one

Todo list:
1. Input and output vocab should be different! 
    • Input vocab should be the whole word2vec vocab, so that similar inputs can be embedded 
        to similar vectors (of course we need to set word2vec matrix's trainble=False). 
    • On the other hand, output vocab should be limited in order to avoid slow softmaxing.
    • When we are doing statistics on output vocab, 
        it should only contain words from target sentences!!
        Note: this point has nothing to do with OpenSubtitles
2. This model cannot work with multi-layer implementation yet. Need to fix that.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from six.moves import xrange
import tensorflow as tf
from pprint import pprint
import sys
import functools
sys.path.append("/home/tongn/hred/")
from src.basic.util import zip_lsts, unzip_lst
from src.basic.metrics import sent_gleu
from src.model.util import (decode, attention, pad_and_truncate, 
                            create_cell, create_MultiRNNCell,
                            bidirecitonal_dynamic_lstm, get_mask,
                            average_gradients, compute_grads, apply_grads, 
                            sequence_loss, double)


# In[2]:


def merge_tensors(tensors, lengths):
    """
    Assuming each element in tensors is 2-D
    """
    merged_length = tf.concat(lengths, axis=0)
    max_len = tf.reduce_max(merged_length)
    padded_tensors = []
    for (tensor, length) in zip(tensors, lengths):
        padded_tensor = pad_and_truncate(tensor, length, max_len)
        padded_tensors.append(padded_tensor)
    merged_tensor = tf.concat(padded_tensors, axis=0)
    return (merged_tensor, merged_length)


# In[3]:


class Seq2seqRL(object):
    """
    An implementation of bi-HRED
    All rights not reserved.
    """
    def __init__(self,
                 batch_size,
                 vocab_size,
                 embedding_size,
                 hidden_size_encoder, hidden_size_decoder,
                 max_iterations,
                 start_token, end_token, unk_indices,
                 num_layers_encoder=1, num_layers_decoder=1,
                 attention_size=512, attention_layer_size=256,
                 beam_width=10, length_penalty_weight=1.0,
                 gpu_start_index=0, 
                 num_gpus=1, # set to 1 when testing
                 learning_rate=0.001, 
                 clipping_threshold=5.0,
                 feed_both_examples=False,
                 use_max_margin=False, max_margin_weight=1.0, margin=0.1,
                 reward_clipping_threshold=1.0,
                 backward=False, # whether we are training a backward model
                 feed_tensors=[], # when provided, placeholders are not used
                 use_MMI_reward=False, MMI_weight=0.00,
                 use_reranking_reward=False, reranking_weight=0.00, num_samples_reranking=2,
                 use_gleu_reward=False, gleu_weight=0.00,
                 softmax_temperature=1.0,
                 beam_search=True): # how many samples to use for baseline (RL training)
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size_encoder = hidden_size_encoder
        self.hidden_size_decoder = hidden_size_decoder
        assert self.hidden_size_encoder * 2 == self.hidden_size_decoder
        self.max_iterations = max_iterations
        self.start_token = start_token
        self.end_token = end_token
        self.unk_indices = unk_indices
        self.num_layers_encoder = num_layers_encoder
        self.num_layers_decoder = num_layers_decoder
        self.attention_size = attention_size
        self.attention_layer_size = attention_layer_size
        self.beam_width = beam_width
        self.length_penalty_weight = length_penalty_weight
        self.gpu_start_index = gpu_start_index
        self.num_gpus = num_gpus
        self.learning_rate = learning_rate
        self.clipping_threshold = clipping_threshold
        self.feed_both_examples = feed_both_examples
        if self.feed_both_examples:
            assert self.batch_size % 2 == 0
            self.half_batch_size = self.batch_size // 2
        self.use_max_margin = use_max_margin
        if self.use_max_margin:
            assert self.feed_both_examples # if Should-Change, then feed_both_examples must be True
        self.max_margin_weight = max_margin_weight
        self.margin = margin
        self.beam_search = beam_search
        
        assert self.batch_size % self.num_gpus == 0
        self.batch_size_per_gpu = self.batch_size // self.num_gpus
        self.feed_tensors = feed_tensors
        
        self.use_MMI_reward = use_MMI_reward
        self.MMI_weight = MMI_weight
        self.use_reranking_reward = use_reranking_reward
        self.reranking_weight = reranking_weight
        self.num_samples_reranking = num_samples_reranking
        self.use_gleu_reward = use_gleu_reward
        self.gleu_weight = gleu_weight
        self.RL_training = self.use_MMI_reward or self.use_reranking_reward or self.use_gleu_reward
        
        self.softmax_temperature = softmax_temperature
        
        if self.feed_both_examples:
            print("Feeding both examples...")
            assert self.batch_size % 2 == 0
            self.norm_batch_size = self.batch_size // 2
        else:
            self.norm_batch_size = self.batch_size # when feeding both examples, only first half are norm inputs
        
        if self.use_max_margin:
            print("Max margin weight: {}, margin: {}".format(self.max_margin_weight, self.margin))
        # We are only performing RL training on the norm_batch_size part, which may or may not be the whole batch
        
        if self.use_MMI_reward:
            print("MMI weight:", self.MMI_weight)
#             self.softmax_temperature = 0.5
#             print("softmax_temperature changed to {}".format(self.softmax_temperature))
        else:
            self.MMI_weight = 0.0
            
        if self.use_reranking_reward:
            print("Neural Reranking reward weight:", self.reranking_weight)
        else:
            self.reranking_weight = 0.0
            
        if self.use_gleu_reward:
            print("GLEU reward weight:", self.gleu_weight)
        else:
            self.gleu_weight = 0.0

        self.ML_weight = 1.0 - (self.MMI_weight + self.reranking_weight + self.gleu_weight)
        
        self.trainable_variables = []
        if self.use_MMI_reward:
            self.trainable_variables_backward = []
        
        # For a backward model, the namespace will be "seq2seq_backward"
        extra_str = "_backward" if backward else ""
        self.main_scope = "seq2seq" + extra_str + "/"
        with tf.device("/gpu:%d" % self.gpu_start_index):
            self.create_placeholders()
            
            # Tile only if we are not training and using beam search
            self.tile = tf.logical_and(
                tf.logical_not(self.is_training), self.beam_search)

            self.global_step = tf.get_variable(
                self.main_scope + "global_step", initializer=0, 
                dtype=tf.int32, trainable=False)

            # Note: if feeding both examples, the first dimension of total_loss is twice as that of 
            # loss_RL's
            (self.total_loss, max_margin_loss, num_tokens,
             loss_MMI, loss_gleu, loss_reranking, num_tokens_RL,
             self.batch_sample_ids_beam, self.batch_final_lengths_beam) = self.one_iteration(
                self.source, self.source_length, 
                self.target, self.target_length, 
                self.start_tokens)

            # This part is for monitoring PPL, not for training.
            if self.feed_both_examples:
                self.batch_num_tokens = num_tokens / 2.0
                # We montior ML losses for both norm- and adv-data
                self.batch_total_loss = (tf.reduce_sum(self.norm(self.total_loss)),
                                         tf.reduce_sum(self.adv(self.total_loss)))
            else:
                self.batch_num_tokens = num_tokens
                self.batch_total_loss = tf.reduce_sum(self.total_loss)

            loss_terms = []
            
            # when using max_margin, it must be a Should-Change strategy
            if self.use_max_margin:
                loss_ML = self.norm(self.total_loss) # in this case we don't want to train on (adv-S, T) pairs
                num_tokens_ML = num_tokens / 2.0
            else:
#                 if self.feed_both_examples:
#                     # This is just for code readability
#                     # We could have just written loss_ML = self.total_loss / 2.0
#                     loss_ML = (self.norm(self.total_loss) + self.adv(self.total_loss)) / 2.0
#                 else:
#                     loss_ML = self.total_loss
                loss_ML = self.total_loss
                num_tokens_ML = num_tokens
                
            loss_terms.append(
                self.ML_weight * tf.reduce_sum(loss_ML) / num_tokens_ML)

            if self.use_max_margin:
                # need to scale max_margin_weight by ML_weight to make the training stable
                # (instead of scaling with loss_ML + loss_RL)
                loss_terms.append(
                    self.max_margin_weight * self.ML_weight * 
#                     tf.reduce_sum(max_margin_loss) / num_tokens_ML)
                    tf.reduce_mean(max_margin_loss))
                
            if self.RL_training and not self.use_max_margin:
                num_tokens_RL = num_tokens_RL / 2.0 # effectively double the RL_loss

            if self.use_MMI_reward:
                loss_terms.append(
                    self.MMI_weight * tf.reduce_sum(loss_MMI) / num_tokens_RL)

            if self.use_reranking_reward:
                loss_terms.append(
                    self.reranking_weight * tf.reduce_sum(loss_reranking) / num_tokens_RL)

            if self.use_gleu_reward:
                loss_terms.append(
                    self.gleu_weight * tf.reduce_sum(loss_gleu) / num_tokens_RL)

            assert loss_terms != []
            loss = tf.add_n(loss_terms)

            optimizer = tf.train.AdamOptimizer(self.learning_rate)            
            grads = compute_grads(
                loss, optimizer, self.trainable_variables)
            self.apply_gradients_op = apply_grads(
                optimizer, [grads], 
                clipping_threshold=self.clipping_threshold, 
                global_step=self.global_step)
    
    def one_iteration(self, 
                      source, source_length, 
                      target, target_length, 
                      start_tokens):        
        self.embedding = self.create_embedding()
        embedding_fn = lambda ids: tf.nn.embedding_lookup(self.embedding, ids)
        output_layer = self.create_output_layer()
        
        with tf.variable_scope(self.main_scope + "encoder", reuse=tf.AUTO_REUSE):
            # Create encoder cells
            [encoder_cell_fw, encoder_cell_bw] = [
                create_cell(self.hidden_size_encoder, self.keep_prob, reuse=tf.AUTO_REUSE)
                for _ in range(2)]
            
            (encoder_output, encoder_final_state) = self.encode(
                encoder_cell_fw, encoder_cell_bw,
                source, source_length,
                embedding_fn)
            
        with tf.variable_scope(self.main_scope + "decoder", reuse=tf.AUTO_REUSE):
            # Create Decoder cell without attention wrapper
            decoder_cell_without_attention = create_cell(
                self.hidden_size_decoder, self.keep_prob, reuse=tf.AUTO_REUSE)
            
            decoder_cell = attention(
                decoder_cell_without_attention, 
                self.attention_size, self.attention_layer_size, 
                self.maybe_tile(encoder_output), 
                self.maybe_tile(source_length))

            decoder_initial_state = self.get_decoder_initial_state(
                decoder_cell, encoder_final_state)

            def get_loss_and_num_tokens(target, target_length):
                """
                Note that batch_loss is only summed along axis=1
                """
                (loss, num_tokens) = self.decode_train(
                    embedding_fn,
                    decoder_cell, decoder_initial_state,
                    start_tokens, target, target_length, 
                    output_layer)
                return (loss, num_tokens)
                        
            (loss_ML, num_tokens) = get_loss_and_num_tokens(target, target_length)
            
            # Get trainable variables
            # (up to now we already have all the seq2seq trainable vars)
            if self.trainable_variables == []:
                self.trainable_variables = tf.get_collection(
                    tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.main_scope)
            
            margin_loss_ground_truth = (self.norm(loss_ML) - self.adv(loss_ML)) / tf.cast(self.norm(target_length), tf.float32)         
            # RL training part: necessary steps for all RL rewards/losses
            if self.RL_training:
                # Sampled sequence
                (logits_sample, sample_ids_sample, final_lengths_sample) = self.decode_sample(
                    embedding_fn,
                    decoder_cell, decoder_initial_state,
                    start_tokens,
                    output_layer,
                    helper_fn=functools.partial(tf.contrib.seq2seq.SampleEmbeddingHelper, 
                                                softmax_temperature=self.softmax_temperature))
                
                # Baseline sequence
                (logits_greedy, sample_ids_greedy, final_lengths_greedy) = self.decode_sample(
                    embedding_fn,
                    decoder_cell, decoder_initial_state,
                    start_tokens,
                    output_layer,
                    helper_fn=tf.contrib.seq2seq.GreedyEmbeddingHelper)

                """
                Improvement: we could sample multiple sequences and only train on the best one
                """                
                # get first part of the raw reward (forward direction)
                (loss_sample, num_tokens_sample) = sequence_loss(
                    logits_sample, sample_ids_sample, final_lengths_sample)
                normalized_loss_sample = loss_sample / tf.cast(final_lengths_sample, tf.float32) # normalize by length
                                
                (loss_greedy, _) = sequence_loss(
                    logits_greedy, sample_ids_greedy, final_lengths_greedy)
                normalized_loss_greedy = loss_greedy / tf.cast(final_lengths_greedy, tf.float32) # normalize by length                
                
                if self.use_max_margin:
                    double_sample_ids_sample = double(self.norm(sample_ids_sample))
                    double_final_lengths_sample = double(self.norm(final_lengths_sample))
                    (loss_sample_max_margin, _) = self.decode_train(
                        embedding_fn,
                        decoder_cell, decoder_initial_state,
                        start_tokens,
                        double_sample_ids_sample[:, :tf.reduce_max(double_final_lengths_sample)], 
                        double_final_lengths_sample,
                        output_layer)
                    # Get max margin loss
                    margin_loss_sample = (self.norm(loss_sample_max_margin) - self.adv(loss_sample_max_margin)) / tf.cast(self.norm(final_lengths_sample), tf.float32)
                    margin_loss = (margin_loss_sample + margin_loss_ground_truth) / 2.0
                    
                # Only keep the norm inputs part of the loss
                # Note: the norm and adv half of num_tokens_sample are different
                [loss_sample, normalized_loss_sample, 
                 num_tokens_sample, 
                 loss_greedy, normalized_loss_greedy] = [
                    self.norm(t) for t
                    in [loss_sample, normalized_loss_sample, 
                        num_tokens_sample, 
                        loss_greedy, normalized_loss_greedy]]
                num_tokens_sample = tf.reduce_sum(num_tokens_sample)
            else:
                if self.use_max_margin:
                    margin_loss = margin_loss_ground_truth
                num_tokens_sample = None
            
            if self.use_max_margin:
                self.avg_margin_loss = tf.reduce_mean(margin_loss) # only works for single GPU                
                max_margin_loss = tf.maximum(0.0, self.margin + margin_loss) # self.margin is broadcast here
            else:
                max_margin_loss = None
                
            
            if self.use_reranking_reward:       
                # Compute basline for loss_RL_dull by investigating what loss will we get on these
                # greedy-decoded utterances if the contexts were different ones
                # We permute greedy/sample-decoded sample_ids and lengths, and then permute the loss back
                # Note: we shouldn't use random.sample because the graph is only built once.
                candidate_offsets = tf.random_shuffle(
                    tf.range(1, limit=self.norm_batch_size - 1, dtype=tf.int32))[:2] # norm_batch_size just in case we are feeding both examples
#                     tf.range(1, limit=self.batch_size - 1, dtype=tf.int32))[:2] # norm_batch_size just in case we are feeding both examples                
                """
                offset tensors (and perm back them) separately for norm- and adv-data
                """
                reranking_loss_sample_perm_lst = []
                reranking_loss_greedy_perm_lst = []
                for i in range(self.num_samples_reranking):
                    offset = candidate_offsets[i]
                    (sample_ids_sample_perm, final_lengths_sample_perm, 
                     sample_ids_greedy_perm, final_lengths_greedy_perm) = [
                        self.offset_tensor(tensor, offset) 
                        for tensor
                        in (sample_ids_sample, final_lengths_sample,
                            sample_ids_greedy, final_lengths_greedy)]
                    
                    (reranking_loss_sample_perm, _) = get_loss_and_num_tokens(
                        sample_ids_sample_perm, final_lengths_sample_perm)
                    (reranking_loss_greedy_perm, _) = get_loss_and_num_tokens(
                        sample_ids_greedy_perm, final_lengths_greedy_perm)
                    
                    def normalize_and_perm_back(reranking_loss_perm, final_lengths_perm):
                        return self.offset_tensor(
                            reranking_loss_perm / tf.cast(final_lengths_perm, tf.float32),
#                             offset=(self.batch_size - offset))
                            offset=(self.norm_batch_size - offset))
                    
                    # Here we only keep the norm part of the loss
                    reranking_loss_sample_perm_lst.append(
                        self.norm(normalize_and_perm_back(reranking_loss_sample_perm, final_lengths_sample_perm)))
                    reranking_loss_greedy_perm_lst.append(
                        self.norm(normalize_and_perm_back(reranking_loss_greedy_perm, final_lengths_greedy_perm)))

                avg_reranking_loss_sample = tf.add_n(reranking_loss_sample_perm_lst) / self.num_samples_reranking
                avg_reranking_loss_greedy = tf.add_n(reranking_loss_greedy_perm_lst) / self.num_samples_reranking
                
                self.avg_reranking_reward = tf.reduce_mean(avg_reranking_loss_sample) # without baseline
                self.reranking_reward = avg_reranking_loss_sample - avg_reranking_loss_greedy
#                 self.avg_reranking_reward = tf.reduce_mean(avg_reranking_loss_sample - normalized_loss_sample) # without baseline
#                 self.reranking_reward = ((avg_reranking_loss_sample - normalized_loss_sample) - 
#                                          (avg_reranking_loss_greedy - normalized_loss_greedy))

                loss_reranking = tf.stop_gradient(self.reranking_reward) * loss_sample
            else:
                loss_reranking = None
                
            if self.use_gleu_reward:
                with tf.device('/cpu:0'):
                    def get_sent_gleu(tup):
                        ((target, target_length), (sample_ids, final_lengths)) = tup
                        ref = target[:target_length]
                        hyp = sample_ids[:final_lengths]
                        gleu = tf.py_func(sent_gleu, [ref, hyp], tf.float32, stateful=False)
                        return gleu

                    gleu_sample = tf.map_fn(
                        get_sent_gleu,
                        ((target[:, :(-1)], target_length - 1), (sample_ids_sample, final_lengths_sample)),
                        dtype=tf.float32)
                    gleu_greedy = tf.map_fn(
                        get_sent_gleu,
                        ((target[:, :(-1)], target_length - 1), (sample_ids_greedy, final_lengths_greedy)),
                        dtype=tf.float32)
                
                gleu_sample = self.norm(gleu_sample)
                gleu_greedy = self.norm(gleu_greedy)
                    
                self.avg_gleu_reward = tf.reduce_mean(gleu_sample)
                self.gleu_reward = gleu_sample - gleu_greedy # reward_greedy is the baseline

                loss_gleu = tf.stop_gradient(self.gleu_reward) * loss_sample
            else:
                loss_gleu = None
        
        # Breaking out of the current scope to load another model
        if self.use_MMI_reward:
            """Get MMI Reward"""
            """
            This part can definitely be simplified by only feeding [:self.norm_batch_size] 
            when self.feed_both_examples == True
            """
            # Merge the sampled sequence and the baseline sequence
            # Note that if feed_both_examples==True, we are only feeding first half of the batch,
            # since RL should only be performed on norm inputs
            (sample_ids_merged, final_lengths_merged) = merge_tensors(
                [self.norm(sample_ids_sample), self.norm(sample_ids_greedy)],
                [self.norm(final_lengths_sample), self.norm(final_lengths_greedy)])
            
            """
            Sing! Fly to prison! Double the guards, double the weapons, 
            double everything. Tailang does not leave that prison!
            """            
            double_source = double(self.norm(self.source))
            double_source_length = double(self.norm(self.source_length))
            double_start_tokens = double(self.norm(self.start_tokens))
                        
            feed_tensors = [
                sample_ids_merged, double_source, # our source is the target in the backward model
                final_lengths_merged, double_source_length,
                tf.constant(1.0), tf.constant(True), # we actually need to use the training branch, otherwise everything will be tiled to batch_size * beam_width
                double_start_tokens]
            max_iterations_backward = tf.reduce_max(double_source_length) + 1 # there is also start_token that will be appended
            
            backward_model = Seq2seqRL(
                self.norm_batch_size * 2, # we are feeding sampled and greedy sequences together, each has norm_batch_size
                self.vocab_size,
                self.embedding_size,
                self.hidden_size_encoder, self.hidden_size_decoder,
                max_iterations_backward, # feeding to max_iterations
                self.start_token, self.end_token, self.unk_indices,
                num_layers_encoder=self.num_layers_encoder, 
                num_layers_decoder=self.num_layers_decoder,
                attention_size=self.attention_size, 
                attention_layer_size=self.attention_layer_size,
                beam_width=self.beam_width, 
                length_penalty_weight=self.length_penalty_weight,
                gpu_start_index=self.gpu_start_index, 
                num_gpus=self.num_gpus,
                learning_rate=self.learning_rate, 
                clipping_threshold=self.clipping_threshold,
                backward=True,
                feed_tensors=feed_tensors)
            
            if self.trainable_variables_backward == []:
                self.trainable_variables_backward = tf.get_collection(
                    tf.GraphKeys.TRAINABLE_VARIABLES, scope=backward_model.main_scope)
            
            # Note that total_loss cannot be changed to batch_total_loss, since the latter is with shape [].
            normallized_loss_backward = backward_model.total_loss / tf.cast(double_source_length, tf.float32)

#             reward_merged = - (tf.concat([normalized_loss_sample, normalized_loss_greedy], axis=0) + normallized_loss_backward)
#             [reward_sample, reward_greedy] = tf.split(reward_merged, 2, axis=0)
            [reward_sample, reward_greedy] = tf.split(normallized_loss_backward, 2, axis=0)
        
            self.avg_MMI_reward = tf.reduce_mean(reward_sample)
            self.MMI_reward = reward_sample - reward_greedy # reward_greedy is the baseline
                        
            loss_MMI = tf.stop_gradient(self.MMI_reward) * loss_sample
        else:
            loss_MMI = None
    
        # Go back to decoder scope
        with tf.variable_scope(self.main_scope + "decoder", reuse=tf.AUTO_REUSE):
            if self.beam_search:
                # Decoder -- beam (for inference)
                decode_fn_infer = self.decode_beam
            else:
                decode_fn_infer = functools.partial(
                    self.decode_sample, helper_fn=tf.contrib.seq2seq.GreedyEmbeddingHelper)
            
            result = decode_fn_infer(
                embedding_fn, decoder_cell, decoder_initial_state,
                start_tokens, output_layer)
            sample_ids = result[-2] # handles both decode_fn
            final_lengths = result[-1]
                
            # pad and truncate
            sample_ids = pad_and_truncate(
                sample_ids, final_lengths, self.max_iterations)
        
        return (loss_ML, max_margin_loss, num_tokens, 
                loss_MMI, loss_gleu, loss_reranking, num_tokens_sample,
                sample_ids, final_lengths)
    
    def create_output_layer(self):
        with tf.variable_scope(self.main_scope + "projection", reuse=tf.AUTO_REUSE):
            output_layer = tf.layers.Dense(
                self.vocab_size,
                use_bias=False,
                trainable=True)
        return output_layer
    
    def maybe_tile(self, tensor):
        tiled_tensor = tf.cond(
            self.tile,
            lambda: tf.contrib.seq2seq.tile_batch(tensor, self.beam_width),
            lambda: tensor)
        return tiled_tensor
    
    def encode(self,
               cell_fw, cell_bw,
               source, source_length, 
               embedding_fn,
               reuse=False):
        with tf.variable_scope("embedding", reuse=tf.AUTO_REUSE):
            embedded_source = embedding_fn(source)
            
        with tf.variable_scope("encoder", reuse=tf.AUTO_REUSE):
            (outputs, final_states) = tf.nn.bidirectional_dynamic_rnn(
                cell_fw, cell_bw, embedded_source,
                sequence_length=source_length,
                dtype=tf.float32, 
                swap_memory=True)
            output = tf.concat(outputs, axis=2)
            (final_state_fw, final_state_bw) = final_states
            final_state = tf.contrib.rnn.LSTMStateTuple(
                tf.concat([final_state_fw.c, final_state_bw.c], axis=1),
                tf.concat([final_state_fw.h, final_state_bw.h], axis=1))

        return (output, final_state)
    
    def get_decoder_initial_state(self, 
                                  decoder_cell,
                                  decoder_initial_state_without_attention=None):
        with tf.variable_scope("decoder", reuse=tf.AUTO_REUSE):            
            decoder_zero_state = tf.cond(
                self.tile,
                lambda: decoder_cell.zero_state(
                    self.batch_size_per_gpu * self.beam_width, tf.float32),
                lambda: decoder_cell.zero_state(self.batch_size_per_gpu, tf.float32))
            # applies to the first turn since there's no context output (i.e., no source utterance) yet
            if decoder_initial_state_without_attention is None: 
                decoder_initial_state = decoder_zero_state
            else:
                decoder_initial_state = decoder_zero_state.clone(
                    cell_state=self.maybe_tile(decoder_initial_state_without_attention))
        return decoder_initial_state        
    
    def decode_train(self,
                     embedding_fn,
                     decoder_cell, decoder_initial_state,
                     start_tokens, target, target_length,
                     output_layer):        
        # Prepend start_token in front of each target
        with tf.variable_scope("embedding", reuse=tf.AUTO_REUSE):
            inputs = tf.concat([tf.expand_dims(start_tokens, axis=1), target[:, :(-1)]], axis=1) # get rid of end_token
            embedded_inputs = embedding_fn(inputs)
        
        # training helper (for teacher forcing)
        with tf.variable_scope("decoder", reuse=tf.AUTO_REUSE):
            helper_train = tf.contrib.seq2seq.TrainingHelper(
                embedded_inputs,
                target_length)
            (decoder_outputs_train, _) = decode(
                decoder_cell, helper_train,
                decoder_initial_state, None, # we don't set max_iter here for training
                output_layer=output_layer)
                
            (logits, _) = decoder_outputs_train            
            decoder_mask_float = self.create_decoder_mask(target, target_length)
            loss = tf.contrib.seq2seq.sequence_loss(
                logits, target, decoder_mask_float,
                average_across_timesteps=False,
                average_across_batch=False,
                name="loss_ML")

            loss_ML = tf.reduce_sum(loss, axis=1)
            num_tokens = tf.reduce_sum(decoder_mask_float)  
        
        return (loss_ML, num_tokens)
    
    def decode_sample(self,
                      embedding_fn,
                      decoder_cell,
                      decoder_initial_state,
                      start_tokens,
                      output_layer,
                      helper_fn=tf.contrib.seq2seq.GreedyEmbeddingHelper):
        helper = helper_fn(
            embedding_fn, start_tokens, self.end_token)
        (decoder_outputs, final_lengths) = decode(
            decoder_cell, helper, decoder_initial_state, 
            self.max_iterations, output_layer=output_layer)
        (logits, sample_ids) = decoder_outputs
        return (logits, sample_ids, final_lengths)
    
    def decode_beam(self, 
                    embedding_fn, 
                    decoder_cell, 
                    decoder_initial_state,
                    start_tokens,
                    output_layer):
        with tf.variable_scope("decoder", reuse=tf.AUTO_REUSE):
            beam_search_decoder = tf.contrib.seq2seq.BeamSearchDecoder(
                decoder_cell, 
                embedding_fn,
                start_tokens, self.end_token,
                decoder_initial_state, self.beam_width,
                output_layer=output_layer,
                length_penalty_weight=self.length_penalty_weight)
            output_beam = tf.contrib.seq2seq.dynamic_decode(
                beam_search_decoder, 
                maximum_iterations=self.max_iterations, 
                swap_memory=True)         
            sample_ids_beam = output_beam[0].predicted_ids[:, :, 0]
            final_lengths_beam = output_beam[2][:, 0]
        return (sample_ids_beam, final_lengths_beam)

    def create_placeholders(self):
        if self.feed_tensors == []:
            self.source = tf.placeholder(
                tf.int32, shape=[self.batch_size, None], name="source")
            self.target = tf.placeholder(
                tf.int32, shape=[self.batch_size, None], name="target")
            
            self.source_length = tf.placeholder(
                tf.int32, shape=[self.batch_size], name="source_length")
            self.target_length = tf.placeholder(
                tf.int32, shape=[self.batch_size], name="target_length")
            
            self.keep_prob = tf.placeholder(
                tf.float32, shape=[], name="keep_prob")
            self.is_training = tf.placeholder(
                tf.bool, shape=[], name="is_training")

            self.start_tokens = tf.placeholder_with_default(
                [self.start_token] * self.batch_size,
                shape=[self.batch_size])
        else:
            [self.source, self.target, 
             self.source_length, self.target_length, 
             self.keep_prob, self.is_training,
             self.start_tokens] = self.feed_tensors

    def create_decoder_mask(self, target, target_length):
        sequence_mask = tf.sequence_mask(target_length)
        unk_mask = get_mask(target, self.unk_indices)
        decoder_mask = tf.logical_and(
            sequence_mask, tf.logical_not(unk_mask))
        decoder_mask_float = tf.cast(decoder_mask, tf.float32)
        return decoder_mask_float
    
    def create_embedding(self):
        with tf.variable_scope(self.main_scope + "embedding"):
            embedding = tf.get_variable(
                "embedding",
                initializer=tf.random_uniform(
                    [self.vocab_size, self.embedding_size], 
                    minval=-0.01, maxval=0.01))
            return embedding
    
    def norm(self, tensor, adv=False):
        if self.feed_both_examples:
#             (norm_half, adv_half) = tf.split(tensor, 2, axis=0)
            if adv:
#                 return adv_half
                return tensor[self.half_batch_size:, ...]
            else:
#                 return norm_half
                return tensor[:self.half_batch_size, ...]
        else:
            return tensor
        
    def adv(self, tensor):
        return self.norm(tensor, adv=True) # when not feeding both examples, this will just return 'tensor'
    
    def offset_tensor(self, tensor, offset):
        """
        Take in a 2-D tensor and offset (permute) it by one position on the first dimension
        Warning: require that the first dimension is at least of size 3 (2 won't work actually)
        Intuition:
            Just like cutting a deck of cards
        """
        if self.feed_both_examples:
            [norm, adv] = tf.split(tensor, 2, axis=0)
            new_tensors = []
            for tensor in [norm, adv]:
                [tensor_1, tensor_2] = tf.split(tensor, [offset, -1], axis=0)
                new_tensor = tf.concat([tensor_2, tensor_1], axis=0)
                new_tensors.append(new_tensor)
            new_tensor = tf.concat(new_tensors, axis=0)
        else:
            [tensor_1, tensor_2] = tf.split(tensor, [offset, -1], axis=0)
            new_tensor = tf.concat([tensor_2, tensor_1], axis=0)

        return new_tensor

