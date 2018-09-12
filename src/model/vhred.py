
# coding: utf-8

# In[1]:


"""
Todo list:
-2. We keep track all the inputs to the context RNN, 
    and apply attention mechanism to those outputs
    if we really want to train decoder on the first two turns,
    it will cause unnecessary calculations, because, e.g.:
        z = tf.multiply(a, b)
        result = tf.cond(x < y, lambda: tf.add(x, z), lambda: tf.square(y))
    If x < y, the tf.add operation will be executed and tf.square operation will not be executed. 
    Since z is needed for at least one branch of the cond, the tf.multiply operation is always executed, 
    unconditionally.
-1. First encode all turns except the last one, then run context-RNN through
all final_state from the encoder, then use decode to collect all the losses
0. See examples/string_split and we should be able to feed in strings
   and train quite well.
1. Input and output vocab should be different! 
    • Input vocab should be the whole word2vec vocab, so that similar inputs can be embedded 
        to similar vectors (of course we need to set word2vec matrix's trainble=False). 
    • On the other hand, output vocab should be limited in order to avoid slow softmaxing.
    • When we are doing statistics on output vocab, 
        it should only contain words from target sentences!!
        Note: this point has nothing to do with OpenSubtitles
3. This model cannot work with multi-layer implementation yet. Need to fix that.
4. have the attention mechanism attend to both context and utterance vectors
                           (i.e., all context vectors so far(optional) 
                                  and the privous (two/all) utterance vectors)
5. When processing 123, keep encoder and context states right after turn 1, and propogate to the next one
6. Have a driver that feed data:
    for each dialogue, keep track of initial states that needs to be fed into graph
7. Need to sort all data based on number of turns 
8. Experiment to be done: can sequence length in dynamic_rnn be all zeros?
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from six.moves import xrange

import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import random
from pprint import pprint
import sys
import functools
from itertools import chain
from src.basic.util import zip_lsts, unzip_lst
from src.model.util import (decode, attention, pad_and_truncate, 
                            create_cell, create_MultiRNNCell,
                            bidirecitonal_dynamic_lstm, get_mask,
                            average_gradients, compute_grads, apply_grads, 
                            convert_to_number, sequence_loss, double)


# In[2]:


class VHRED(object):
    """
    An implementation of bi-HRED
    All rights not reserved.
    """
    def __init__(self,
                 batch_size,
                 vocab_size,
                 embedding_size,
                 hidden_size_encoder, hidden_size_context, hidden_size_decoder,
                 max_iterations, max_dialogue_length,
                 start_token, end_token, unk_indices,
                 num_layers_encoder=1, num_layers_context=1, num_layers_decoder=1,
                 attention_size=512, attention_layer_size=256,
                 beam_search=False, beam_width=10, length_penalty_weight=1.0,
                 gpu_start_index=0, 
                 learning_rate=0.001, 
                 clipping_threshold=5.0,
                 feed_both_examples=False,
                 use_max_margin=False, max_margin_weight=1.0, margin=0.5):
        self.batch_size = batch_size
        self.half_batch_size = self.batch_size // 2
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size_encoder = hidden_size_encoder
        self.hidden_size_context = hidden_size_context
        self.hidden_size_decoder = hidden_size_decoder
        self.dim_z = self.hidden_size_context # this decision is arbitrary
        
        self.create_context_initial_state_var = functools.partial(
            tf.get_variable, 
            initializer=tf.zeros([self.batch_size, self.hidden_size_context]),
            dtype=tf.float32, trainable=False)
        
        self.dense = functools.partial(tf.layers.dense, units=self.dim_z, use_bias=True)
        
        self.max_iterations = max_iterations
        self.max_dialogue_length = max_dialogue_length
        assert self.max_dialogue_length > 0
        
        self.start_token = start_token
        self.end_token = end_token
        self.unk_indices = unk_indices
        self.num_layers_encoder = num_layers_encoder
        self.num_layers_context = num_layers_context
        self.num_layers_decoder = num_layers_decoder
        self.attention_size = attention_size
        self.attention_layer_size = attention_layer_size
        self.beam_search = beam_search
        self.beam_width = beam_width
        self.length_penalty_weight = length_penalty_weight
        self.clipping_threshold = clipping_threshold
        self.feed_both_examples = feed_both_examples
        if self.feed_both_examples:
            assert self.batch_size % 2 == 0
        self.use_max_margin = use_max_margin
        if self.use_max_margin:
            assert self.feed_both_examples
        self.max_margin_weight = max_margin_weight
        self.margin = margin
            
        self.prior_str = "prior"
        self.posterior_str = "posterior"
        """
        context_input_acc put context inputs in reverse order.
            • When adding inputs, just concat from left
            • When using, we do not need to reverse it back because we are using bidirectional-lstm
                Note: If we don't use bidirectional-lstm, then we will need tf.reverse_sequence()
        """
        self.trainable_variables = []
        
        with tf.variable_scope("seq2seq"):
            self.create_placeholders()

            # Tile only if we are not training and using beam search
            self.tile = tf.logical_and(
                tf.logical_not(self.is_training), self.beam_search)

            context_input_acc = tf.get_variable(
                "context_input_acc",
                initializer=tf.zeros([self.batch_size, 2, self.hidden_size_encoder * 2], dtype=tf.float32),
                trainable=False)
            self.global_step = tf.get_variable(
                "global_step", initializer=0, dtype=tf.int32,
                trainable=False)

            # max_num_turns better not be less than 2, since we may just lose a whole dimension (i.e., axis=1)?
            max_num_turns = tf.maximum(tf.reduce_max(self.start_turn_index), 2)
            context_input_mask = tf.tile(
                tf.reshape(tf.greater(self.start_turn_index, 0), [self.batch_size, 1, 1]), # expand two dims
                [1, max_num_turns, self.hidden_size_encoder * 2])
            # This multiplication resets context input that have start_turn_index == 0
            previous_context_input = context_input_acc[:, :max_num_turns, :] * tf.cast(context_input_mask, tf.float32)

            optimizer = tf.train.AdamOptimizer(learning_rate)

            with tf.device("/gpu:%d" % gpu_start_index):                    
                (kl_loss, total_loss, num_tokens, max_margin_loss, 
                 context_input, sample_ids, final_lengths) = self.one_iteration(
                    self.dialogue, self.turn_length, 
                    previous_context_input, 
                    self.start_turn_index, self.start_tokens)

                kl_loss_weight = tf.cond(
                    self.tile,
                    lambda: tf.minimum(1.0 / 75000.0 * tf.cast(self.global_step, tf.float32), 1.0),
                    lambda: 1.0)
                if self.use_max_margin:
                    kl_loss = kl_loss[:self.half_batch_size]
                weighted_kl_loss = kl_loss_weight * tf.reduce_mean(kl_loss)

                if self.use_max_margin:                        
                    num_tokens = num_tokens / 2.0
                    # Note: here total_loss is a 1-D vector (already summed with axis=1)
                    cross_ent_loss = tf.reduce_sum(total_loss[:self.half_batch_size])
                    adv_cross_ent_loss = tf.reduce_sum(total_loss[self.half_batch_size:])
                    self.batch_total_loss = (cross_ent_loss, adv_cross_ent_loss)

                    loss = (weighted_kl_loss + cross_ent_loss / num_tokens 
                            + self.max_margin_weight * tf.reduce_mean(max_margin_loss)) # max_margin_loss shape=[self.half_batch_size]
                else:
                    cross_ent_loss = tf.reduce_sum(total_loss)
                    self.batch_total_loss = cross_ent_loss
                    loss = weighted_kl_loss + cross_ent_loss / num_tokens
                
                self.batch_num_tokens = num_tokens
                
                grads = compute_grads(
                    loss, optimizer, self.trainable_variables)
                
                # First context input will be repeated in the next batch, so we ignore it.
                assign_context_input_op = tf.assign(
                    context_input_acc, context_input[:, 1:, :], validate_shape=False) # shape will be differnt on axis=1
            
                with tf.control_dependencies([assign_context_input_op]): # make sure we update context_input_acc
                    self.apply_gradients_op = apply_grads(
                        optimizer, [grads], 
                        clipping_threshold=self.clipping_threshold, 
                        global_step=self.global_step)

                    # Just for control dependencies
                    self.batch_sample_ids_beam = tf.identity(sample_ids)
                    self.batch_final_lengths_beam = tf.identity(final_lengths)
    
    def one_iteration(self, 
                      dialogue, turn_length,
                      previous_context_input, 
                      start_turn_index, start_tokens):
        turns = [convert_to_number(dialogue[:, i]) 
                 for i in range(self.max_dialogue_length)]
        turn_lengths = [turn_length[:, i] for i in range(self.max_dialogue_length)]
        
        embedding = self.create_embedding()
        self.embedding = embedding # just for retrieving the embedding
        output_layer = self.create_output_layer()
        
        with tf.variable_scope("encoder", reuse=tf.AUTO_REUSE):
            # Create encoder cells
            [encoder_cell_fw, encoder_cell_bw] = [
                create_cell(self.hidden_size_encoder, self.keep_prob, reuse=tf.AUTO_REUSE)
                for _ in range(2)]
            
            encoder_final_state_h_lst = []
            # Iterate through all turns (last turn is needed for posterior parameterization)
            for i in xrange(self.max_dialogue_length):
                encoder_input = turns[i]
                encoder_length = turn_lengths[i]
                (encoder_output, encoder_final_state_h) = self.encode(
                    encoder_cell_fw, encoder_cell_bw,
                    encoder_input, encoder_length,
                    embedding)
                encoder_final_state_h_lst.append(encoder_final_state_h)
            
            # need all turns except the last one
            stacked_encoder_final_state_h = tf.stack(encoder_final_state_h_lst[:(-1)], axis=1)
            context_input = tf.concat(
                [tf.reverse(stacked_encoder_final_state_h, axis=[1]), # our previous_context_input is backwards
                 previous_context_input],
                axis=1)

        # Create context cell
        with tf.variable_scope("context", reuse=tf.AUTO_REUSE):
            [context_cell_fw, context_cell_bw] = [
                create_cell(self.hidden_size_context, self.keep_prob, reuse=tf.AUTO_REUSE)
                for _ in range(2)]
            (context_outputs, context_final_states) = tf.nn.bidirectional_dynamic_rnn(
                context_cell_fw, context_cell_bw, 
                context_input,
                sequence_length=start_turn_index + self.max_dialogue_length - 1, # last turn is not in context
                dtype=tf.float32, 
                swap_memory=True)
            context_output = tf.concat(context_outputs, axis=2)
            (context_final_state_fw, context_final_state_bw) = context_final_states
            context_final_state_h_concat = tf.concat(
                [context_final_state_fw.h, context_final_state_bw.h], 
                axis=1)
        
        with tf.variable_scope("latent", reuse=tf.AUTO_REUSE):
            (kl_loss, sample_z) = self.get_kl_loss_and_sample(
                context_output[:, -1, :], # we already know that there are no paddings in context-level encoding
                encoder_final_state_h_lst[-1])
            if self.feed_both_examples: # in this case we only need kl_loss of normal input
                kl_loss = kl_loss[:self.half_batch_size]
            decoder_input = tf.concat([context_final_state_h_concat, sample_z], axis=1)
            
            def decoder_embedding_fn(ids, is_2d=True, beam_search=False):
                """
                Args:
                    ids: rank=1 or 2
                Ref: https://stackoverflow.com/questions/36041171/tensorflow-concat-a-variable-sized-placeholder-with-a-vector
                (is_2d, beam_search) = 
                    (True, False) training
                    (True, True) beam search
                    (False, False) greedy decoding inference
                """
                embedded_ids = self.embed(embedding, ids)
                
                if is_2d:
                    if beam_search:
                        len_tile = self.beam_width
                    else:
                        ids_shape = tf.shape(ids)
                        len_tile = ids_shape[1]
                    tiled_decoder_input = tf.tile(
                        tf.expand_dims(decoder_input, axis=1), 
                        [1, len_tile, 1])
                else:
                    tiled_decoder_input = decoder_input
                
                # Handles ids of both rank 1 & 2
                embedded_concat = tf.concat([embedded_ids, tiled_decoder_input], axis=-1)

                return embedded_concat
            
        with tf.variable_scope("decoder", reuse=tf.AUTO_REUSE):
            # Create Decoder cell without attention wrapper
            decoder_cell_without_attention = create_cell(
                self.hidden_size_decoder, self.keep_prob, reuse=tf.AUTO_REUSE)
            
            memory = context_output
            memory_length = start_turn_index + self.max_dialogue_length - 1 # broadcasting self.max_dialogue_length
                
            decoder_cell = attention(
                decoder_cell_without_attention, 
                self.attention_size, self.attention_layer_size, 
                self.maybe_tile(memory), 
                self.maybe_tile(memory_length))
            
            decoder_initial_state = self.get_decoder_initial_state(
                decoder_cell, decoder_initial_state_without_attention=None)

            # Note that batch_loss is only summed along axis=1
            (loss_ML, num_tokens) = self.decode_train(
                decoder_embedding_fn,
                decoder_cell, decoder_initial_state,
                start_tokens, turns[-1], turn_lengths[-1],
                output_layer)       

            # Get trainable variables
            # (up to now we already have all the seq2seq trainable vars)
            if self.trainable_variables == []:
                self.trainable_variables = tf.get_collection(
                    tf.GraphKeys.TRAINABLE_VARIABLES, scope="seq2seq")

            # Decoder -- beam (for inference)
            embedding_fn = functools.partial(
                decoder_embedding_fn, is_2d=self.beam_search, 
                beam_search=self.beam_search) # both determined by self.beam_search
            
            if self.beam_search:
                decode_fn_infer = self.decode_beam
            else:
                decode_fn_infer = functools.partial(
                    self.decode_sample, helper_fn=tf.contrib.seq2seq.GreedyEmbeddingHelper)
            
            if self.use_max_margin:
                decode_fn_max_margin = functools.partial(
                        self.decode_sample, helper_fn=tf.contrib.seq2seq.SampleEmbeddingHelper)
                decode_outputs = tf.cond(
                    self.is_training,
                    lambda: decode_fn_infer(embedding_fn, decoder_cell, 
                                            decoder_initial_state, output_layer),
                    lambda: decode_fn_max_margin(embedding_fn, decoder_cell, 
                                                 decoder_initial_state, output_layer))
            else:
                decode_outputs = decode_fn_infer(
                    embedding_fn, decoder_cell, 
                    decoder_initial_state, output_layer)

            if self.beam_search:
                (sample_ids, final_lengths) = decode_outputs
            else:
                (logits, sample_ids, final_lengths) = decode_outputs
            
            if self.use_max_margin:
                double_sample_ids = double(self.norm(sample_ids))
                double_final_lengths = double(self.norm(final_lengths))
                
                (loss_sample_max_margin, _) = self.decode_train(
                    decoder_embedding_fn,
                    decoder_cell, decoder_initial_state,
                    start_tokens,
                    double_sample_ids[:, :tf.reduce_max(double_final_lengths)],
                    double_final_lengths,
                    output_layer)
                
                self.loss_sample_max_margin = loss_sample_max_margin
                
                # Get max margin loss
#                 margin_loss = (loss_ML[:self.half_batch_size] - loss_ML[self.half_batch_size:]) / tf.cast(turn_lengths[-1][:self.half_batch_size], tf.float32)
                margin_loss = (loss_sample_max_margin[:self.half_batch_size] - loss_sample_max_margin[self.half_batch_size:]) / tf.cast(final_lengths[:self.half_batch_size], tf.float32)
                self.avg_margin_loss = tf.reduce_mean(margin_loss)                
                max_margin_loss = tf.maximum(0.0, self.margin + margin_loss) # self.margin is broadcast here
            else:
                max_margin_loss = None
            
        # pad and truncate
        sample_ids = pad_and_truncate(
            sample_ids, final_lengths, self.max_iterations)
            
        return (kl_loss, loss_ML, num_tokens, max_margin_loss, 
                context_input, 
                sample_ids, 
                final_lengths)
    
    def embed(self, embedding, ids):
        embedded_ids = tf.nn.embedding_lookup(embedding, ids)
        return embedded_ids
    
    def create_output_layer(self):
        with tf.variable_scope("projection", reuse=tf.AUTO_REUSE):
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
    
    def get_kl_loss_and_sample(self,
                               context_output_last_turn, 
                               encoder_final_state_h_last_turn):
        # Get prior distribution
        h_bar_t_con = self.get_h_bar(
            context_output_last_turn, self.prior_str)
        mu_prior = self.get_mu(h_bar_t_con, self.prior_str)
        sigma_prior = self.get_sigma(h_bar_t_con, self.prior_str)
        dist_prior = tfp.distributions.MultivariateNormalDiag(
            loc=mu_prior, scale_diag=sigma_prior)

        # Get posterior distribution
        h_t_p = tf.concat(
            [context_output_last_turn, 
             encoder_final_state_h_last_turn],
            1) # 0th dimension is batch_size
        h_bar_t_p = self.get_h_bar(h_t_p, self.posterior_str)
        mu_posterior = self.get_mu(h_bar_t_p, self.posterior_str)
        sigma_posterior = self.get_sigma(h_bar_t_p, self.posterior_str)
        dist_posterior = tfp.distributions.MultivariateNormalDiag(
            loc=mu_posterior, scale_diag=sigma_posterior)

        # Calculate KL divergence
        kl_loss = tf.distributions.kl_divergence(dist_prior, dist_posterior)

        # Get distribution based on whether we are training
        (mu, sigma) = tf.cond(
            self.is_training,
            lambda: (mu_posterior, sigma_posterior), # when training, we use posterior distribution
            lambda: (mu_prior, sigma_prior)) # when not training, we use prior distribution
        dist = tfp.distributions.MultivariateNormalDiag(
            loc=mu, scale_diag=sigma)

        sample_z = dist.sample()
        
        return (kl_loss, sample_z)
    
    def get_h_bar(self, h, scope):
        """
        Args:
            scope: "prior" or "posterior"
        """
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            with tf.variable_scope("layer_1", reuse=tf.AUTO_REUSE):
                result_1 = tf.tanh(self.dense(h))
            with tf.variable_scope("layer_2", reuse=tf.AUTO_REUSE):
                h_bar = tf.tanh(self.dense(result_1))
        return h_bar
    
    def get_mu(self, h_bar, scope):
        """
        Args:
            scope: "prior" or "posterior"
        """
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            with tf.variable_scope("mu", reuse=tf.AUTO_REUSE):
                mu = self.dense(h_bar)
        return mu
    
    def get_sigma(self, h_bar, scope):
        """
        Args:
            scope: "prior" or "posterior"
        """
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            with tf.variable_scope("sigma", reuse=tf.AUTO_REUSE):
                # multiply with 0.1 to make training more stable
                sigma = 0.1 * tf.log(1 + tf.exp(1 + self.dense(h_bar)))# don't need tf.diag() here
        return sigma
    
    def get_decoder_initial_state(self, decoder_cell,
                                  decoder_initial_state_without_attention=None):
        with tf.variable_scope("decoder", reuse=tf.AUTO_REUSE):            
            decoder_zero_state = tf.cond(
                self.tile,
                lambda: decoder_cell.zero_state(
                    self.batch_size * self.beam_width, tf.float32),
                lambda: decoder_cell.zero_state(self.batch_size, tf.float32))
            # applies to the first turn since there's no context output (i.e., no source utterance) yet
            if decoder_initial_state_without_attention is None: 
                decoder_initial_state = decoder_zero_state
            else:
                decoder_initial_state = decoder_zero_state.clone(
                    cell_state=self.maybe_tile(decoder_initial_state_without_attention))
        return decoder_initial_state        
    
    def decode_sample(self,
                      embedding_fn, 
                      decoder_cell, 
                      decoder_initial_state, 
                      output_layer,
                      helper_fn=tf.contrib.seq2seq.GreedyEmbeddingHelper):
        helper = helper_fn(
            embedding_fn, self.start_tokens, self.end_token)
        (decoder_outputs, final_lengths) = decode(
            decoder_cell, helper, decoder_initial_state, 
            self.max_iterations, output_layer=output_layer)
        (logits, sample_ids) = decoder_outputs
        
        return (logits, sample_ids, final_lengths)
    
    def decode_beam(self, 
                    embedding_fn, 
                    decoder_cell, 
                    decoder_initial_state, 
                    output_layer):
        with tf.variable_scope("decoder", reuse=tf.AUTO_REUSE):
            beam_search_decoder = tf.contrib.seq2seq.BeamSearchDecoder(
                decoder_cell, 
                embedding_fn,
                self.start_tokens, self.end_token,
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
                logits,
                target,
                decoder_mask_float,
                average_across_timesteps=False,
                average_across_batch=False,
                name="loss_ML")

            loss_ML = tf.reduce_sum(loss, axis=1)
            num_tokens = tf.reduce_sum(decoder_mask_float)  
        
        return (loss_ML, num_tokens)
    
    def encode(self,
               cell_fw, cell_bw,
               inputs, sequence_length, 
               embedding,
               reuse=False):
        with tf.variable_scope("embedding", reuse=tf.AUTO_REUSE):
            embedded_inputs = self.embed(embedding, inputs)
            
        with tf.variable_scope("encoder", reuse=tf.AUTO_REUSE):
            (outputs, final_states) = tf.nn.bidirectional_dynamic_rnn(
                cell_fw, cell_bw, embedded_inputs,
                sequence_length=sequence_length,
                dtype=tf.float32, 
                swap_memory=True)
                        
            output = tf.concat(outputs, axis=2)
            final_state_h = tf.concat(
                [final_states[0].h, final_states[1].h],
                axis=1)

        return (output, final_state_h)

    def create_placeholders(self):
        self.dialogue = tf.placeholder(
            tf.string, [self.batch_size, self.max_dialogue_length], name="dialogue")
        self.turn_length = tf.placeholder(
            tf.int32, [self.batch_size, self.max_dialogue_length], name="utterance_length")
        
        self.keep_prob = tf.placeholder(
            tf.float32, shape=[], name="keep_prob")
        self.is_training = tf.placeholder(
            tf.bool, shape=[], name="is_training")

        self.start_turn_index = tf.placeholder(
            tf.int32, shape=[self.batch_size], name="start_turn_index")
        self.start_tokens = tf.placeholder_with_default(
            [self.start_token] * self.batch_size,
            shape=[self.batch_size])

    def create_decoder_mask(self, target, target_lengths):
        sequence_mask = tf.sequence_mask(target_lengths)
        unk_mask = get_mask(target, self.unk_indices)
        decoder_mask = tf.logical_and(
            sequence_mask, tf.logical_not(unk_mask))
        decoder_mask_float = tf.cast(decoder_mask, tf.float32)
        return decoder_mask_float
    
    def create_embedding(self):
        with tf.variable_scope("embedding"):
            embedding = tf.get_variable(
                "embedding",
                initializer=tf.random_uniform(
                    [self.vocab_size, self.embedding_size], 
                    minval=-0.01, maxval=0.01))
        return embedding
        
    @staticmethod
    def concat_turns(tup):
        """
        tensors: rank=2
        lengths: rank=1
        """
        (tensors, lengths) = tup
        actual_tensors = [tensor[:length, :] for (tensor, length) in zip(tensors, lengths)]
        paddings = [tensor[length:, :] for (tensor, length) in zip(tensors, lengths)]
        tensors_concat = tf.concat(actual_tensors + paddings, 0) # concat on the time_step dimension
        return tensors_concat
    
    def norm(self, tensor, adv=False):
        if self.feed_both_examples:
            if adv:
                return tensor[self.half_batch_size:, ...]
            else:
                return tensor[:self.half_batch_size, ...]
        else:
            return tensor
        
    def adv(self, tensor):
        return self.norm(tensor, adv=True) # when not feeding both examples, this will just return 'tensor'

