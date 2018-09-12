
# coding: utf-8

# In[2]:


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from six.moves import xrange
import tensorflow as tf
import sys
import functools
from pprint import pprint
from src.basic.util import zip_lsts, unzip_lst
from src.model.util import (decode, attention, pad_and_truncate, 
                            create_cell_with_dropout,
                            create_cell, create_MultiRNNCell, attention,
                            bidirecitonal_dynamic_lstm, get_mask,
                            average_gradients, compute_grads, apply_grads, 
                            tile_single_cell_state, tile_multi_cell_state)


# In[2]:


class HRED(object):
    """
    An implementation of bi-HRED
    """
    def __init__(self,
                 batch_size,
                 vocab_size,
                 embedding_size,
                 hidden_size_encoder, hidden_size_context, hidden_size_decoder,
                 dim_z,
                 max_iterations, max_dialogue_length,
                 start_token, end_token, unk_indices,
                 num_layers_encoder=1, num_layers_context=1, num_layers_decoder=1,
                 attention_size=512, attention_layer_size=256,
                 beam_width=10, length_penalty_weight=1.0,
                 gpu_start_index=0, 
                 num_gpus=1, # set to 1 when testing
                 learning_rate=0.001, 
                 clipping_threshold=5.0,
                 truncated=True):
        self.batch_size = batch_size
        self.batch_size_per_gpu = batch_size // num_gpus
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size_encoder = hidden_size_encoder
        self.hidden_size_context = hidden_size_context
        self.hidden_size_decoder = hidden_size_decoder
        assert self.hidden_size_context * 2 == self.hidden_size_decoder
        
        self.dim_z = dim_z
        self.dense = functools.partial(tf.layers.dense, units=self.dim_z, use_bias=True)
        
        self.max_iterations = max_iterations
        self.max_dialogue_length = max_dialogue_length
        assert self.max_dialogue_length > 0
        
        self.start_tokens = [start_token] * self.batch_size_per_gpu
        self.end_token = end_token
        self.unk_indices = unk_indices
        self.num_layers_encoder = num_layers_encoder
        self.num_layers_context = num_layers_context
        self.num_layers_decoder = num_layers_decoder
        self.attention_size = attention_size
        self.attention_layer_size = attention_layer_size
        self.beam_width = beam_width
        self.length_penalty_weight = length_penalty_weight
        self.num_gpus = num_gpus
        self.clipping_threshold = clipping_threshold
        self.truncated = truncated
        """
        context_input_acc put context inputs in reverse order.
            • When adding inputs, just concat from left
            • When using, we do not need to reverse it back because we are using bidirectional-lstm
                Note: If we don't use bidirectional-lstm, then we will need tf.reverse_sequence()
        """
        self.trainable_variables = []
        
        with tf.variable_scope("seq2seq"):
            with tf.device('/cpu:0'):
                self.create_placeholders()
                
                context_input_acc = tf.get_variable(
                    "context_input_acc",
                    initializer=tf.zeros([self.batch_size, 2, self.hidden_size_encoder * 2], dtype=tf.float32),
                    trainable=False)
                
                # max_num_turns better not be less than 2, since we may just lose a whole dimension (i.e., axis=1)?
                max_num_turns = tf.maximum(tf.reduce_max(self.start_turn_index), 2)
                context_input_mask = tf.tile(
                    tf.reshape(tf.greater(self.start_turn_index, 0), [self.batch_size, 1, 1]), # expand two dims
                    [1, max_num_turns, self.hidden_size_encoder * 2])
                # This multiplication resets context input that have start_turn_index == 0
                previous_context_input = context_input_acc[:, :max_num_turns, :] * tf.cast(context_input_mask, tf.float32)
 
                # Note: Make sure batch_size can be evenly divided by num_gpus
                [dialogue_lst, turn_length_lst,
                 previous_context_input_lst, 
                 start_turn_index_lst] = [
                    tf.split(tensor, self.num_gpus, axis=0)
                    for tensor 
                    in [self.dialogue, self.turn_length,
                        previous_context_input, #cannot be less than 2, otherwise tf.map_fn will give error.
                        self.start_turn_index]]

                optimizer = tf.train.AdamOptimizer(learning_rate)

                context_input_lst = []
                sample_ids_beam_lst = []
                final_lengths_beam_lst = []
                num_tokens_lst = []
                total_losses = []
                tower_grads = []
            for i in xrange(num_gpus):
                with tf.device("/gpu:%d" % (gpu_start_index + i)):                    
                    (total_loss, num_tokens, context_input,
                     sample_ids_beam, final_lengths_beam) = self.one_iteration(
                        dialogue_lst[i], turn_length_lst[i], 
                        previous_context_input_lst[i], start_turn_index_lst[i],
                        optimizer)
                    
                    # first turn will be repeated in the next batch, so we skip it
                    context_input_lst.append(context_input[:, 1:, :])
                    
                    sample_ids_beam_lst.append(sample_ids_beam)
                    final_lengths_beam_lst.append(final_lengths_beam)
                    
                    grads = compute_grads(
                        total_loss / num_tokens, optimizer, self.trainable_variables)
                    tower_grads.append(grads)
                    
                    total_losses.append(total_loss)
                    num_tokens_lst.append(num_tokens)
                    
            with tf.device('/cpu:0'):
                context_input_concat = tf.concat(context_input_lst, axis=0)
                assign_context_input_op = tf.assign(
                    context_input_acc, context_input_concat, validate_shape=False) # shape will be differnt on axis=1
                with tf.control_dependencies([assign_context_input_op]): # make sure we update context_input_acc
                    # Concat sample ids and their respective lengths
                    self.batch_sample_ids_beam = tf.concat(
                        sample_ids_beam_lst, axis=0)
                    self.batch_final_lengths_beam = tf.concat(
                        final_lengths_beam_lst, axis=0)

                    self.batch_total_loss = tf.add_n(total_losses)
                    self.batch_num_tokens = tf.add_n(num_tokens_lst)

                    self.apply_gradients_op = apply_grads(
                        optimizer, tower_grads, clipping_threshold=self.clipping_threshold)
    
    def one_iteration(self, 
                      dialogue, turn_length,
                      previous_context_input, start_turn_index,
                      optimizer):
        turns = [HRED.convert_to_number(dialogue[:, i]) for i in range(self.max_dialogue_length)]
        turn_lengths = [turn_length[:, i] for i in range(self.max_dialogue_length)]
        
        embedding = self.create_embedding()
        output_layer = self.create_output_layer(embedding)
        
        with tf.variable_scope("encoder", reuse=tf.AUTO_REUSE):
            # Create encoder cells
            [encoder_cell_fw, encoder_cell_bw] = [
                create_cell_with_dropout(self.hidden_size_encoder, self.keep_prob, reuse=tf.AUTO_REUSE)
                for _ in range(2)]
            
            encoder_final_state_h_lst = []
            # Iterate through all turns except the last one
            for i in xrange(self.max_dialogue_length - 1):
                encoder_input = turns[i]
                encoder_length = turn_lengths[i]
                (_, encoder_final_state_h) = self.encode(
                    encoder_cell_fw, encoder_cell_bw,
                    encoder_input, encoder_length,
                    embedding)
                
                encoder_final_state_h_lst.append(encoder_final_state_h)

            stacked_encoder_final_state_h = tf.stack(encoder_final_state_h_lst, axis=1)
            context_input = tf.concat(
                [tf.reverse(stacked_encoder_final_state_h, axis=[1]), # our previous_context_input is backwards
                 previous_context_input],
                axis=1)

        # Create context cell
        with tf.variable_scope("context", reuse=tf.AUTO_REUSE):
            [context_cell_fw, context_cell_bw] = [
                create_cell_with_dropout(self.hidden_size_context, self.keep_prob, reuse=tf.AUTO_REUSE)
                for _ in range(2)]
            (context_outputs, context_final_states) = tf.nn.bidirectional_dynamic_rnn(
                context_cell_fw, context_cell_bw, 
                context_input,
                sequence_length=start_turn_index + self.max_dialogue_length - 1, # last turn is not in context
                dtype=tf.float32, 
                swap_memory=True)
            context_output = tf.concat(context_outputs, axis=2)
            (context_final_state_fw, context_final_state_bw) = context_final_states
            context_final_state_concat = tf.contrib.rnn.LSTMStateTuple(
                tf.concat([context_final_state_fw.c, context_final_state_bw.c], axis=1),
                tf.concat([context_final_state_fw.h, context_final_state_bw.h], axis=1))
            context_final_state = context_final_state_concat
            
        with tf.variable_scope("decoder", reuse=tf.AUTO_REUSE):
            # Create Decoder cell without attention wrapper
            decoder_cell_without_attention = create_cell_with_dropout(
                self.hidden_size_decoder, self.keep_prob, reuse=tf.AUTO_REUSE)
            decoder_cell = attention(
                decoder_cell_without_attention, 
                self.attention_size, self.attention_layer_size, 
                self.tile_if_not_training(context_output), 
                self.tile_if_not_training(start_turn_index + self.max_dialogue_length - 1)) # broadcasting self.max_dialogue_length
#             decoder_cell = tf.contrib.rnn.OutputProjectionWrapper(
#                 decoder_cell, self.vocab_size, reuse=tf.AUTO_REUSE)
            
            decoder_initial_state = self.get_decoder_initial_state(
                decoder_cell, 
                context_final_state=context_final_state, 
                reuse=tf.AUTO_REUSE)
            
            if self.truncated: # if it is the truncated version, them start_token is actually "__eot__"
                target = turns[-1]
                target_length = turn_lengths[-1]
            else:            
                # Prepend start_token in front of each target, and adjust target_lengths
                target = tf.concat(
                    [tf.expand_dims(tf.constant(self.start_tokens), axis=1), turns[-1]],
                    axis=1)
                target_length = turn_lengths[-1] + 1
            
            # Note that batch_loss is only summed along axis=1
            (loss, num_tokens) = self.decode_train(
                target, target_length, embedding,
                decoder_cell, decoder_initial_state, 
                output_layer)

            # Get trainable variables
            # (up to now we already have all the seq2seq trainable vars)
            if self.trainable_variables == []:
                self.trainable_variables = tf.get_collection(
                    tf.GraphKeys.TRAINABLE_VARIABLES, scope="seq2seq")

            # Decoder -- beam (for inference)
            (sample_ids_beam, final_lengths_beam) = self.decode_beam(
                embedding, decoder_cell, 
                decoder_initial_state, 
                output_layer)
            
            # pad and truncate
            sample_ids_beam = pad_and_truncate(
                sample_ids_beam, final_lengths_beam, self.max_iterations)
            
        return (loss, num_tokens, context_input,
                sample_ids_beam, final_lengths_beam)
    
    def create_output_layer(self, embedding):
        with tf.variable_scope("projection", reuse=tf.AUTO_REUSE):
            output_layer = tf.layers.Dense(
                self.vocab_size,
                use_bias=False,
                kernel_initializer=tf.glorot_uniform_initializer,
                trainable=True)
        return output_layer
    
    def tile_if_not_training(self, tensor):
        tiled_tensor = tf.cond(
            self.is_training, 
            lambda: tensor,
            lambda: tf.contrib.seq2seq.tile_batch(tensor, self.beam_width))
        return tiled_tensor

    @staticmethod
    def convert_to_number(string_tensor):
        """
        string_tensor: a 1-D string tensor
        """
        tokenized = tf.sparse_tensor_to_dense(
            tf.string_split(string_tensor, delimiter=' ', skip_empty=True),
            default_value='0')
        number = tf.string_to_number(tokenized, out_type=tf.int32)
        return number
    
    @staticmethod
    def get_kl_loss(context_outputs, encoder_final_states):
        """
        Todo: should we average kl_loss and ML loss over num_turns in a dialogue?
        """
        kl_lst = []
        for (context_output, encoder_final_state) in zip(context_outputs, encoder_final_states):
            # Get prior distribution
            dist_prior = get_dist(context_output, "prior")
            # Get posterior distribution
            h_t_p = tf.concat([context_output, encoder_final_state], 1) # 0th dimension is batch_size
            dist_posterior = get_dist(h_t_p, "posterior")
            kl = tf.distributions.kl_divergence(dist_prior, dist_posterior)
            kl_lst.append(kl)
        kl_loss = tf.add_n(kl_lst)
        return kl_loss
    
    def get_dist(h, scope):
        mu = get_mu(h, scope)
        sigma = get_sigma(h, scope)
        dist = tf.contrib.distributions.MultivariateNormalDiag(loc=mu, scale_diag=sigma)
        return dist
    
    def get_h_bar(h, scope):
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
    
    def get_mu(h_bar, scope):
        """
        Args:
            scope: "prior" or "posterior"
        """
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            with tf.variable_scope("mu", reuse=tf.AUTO_REUSE):
                mu = self.dense(h_bar)
        return mu
    
    def get_sigma(h_bar, scope):
        """
        Args:
            scope: "prior" or "posterior"
        """
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            with tf.variable_scope("sigma", reuse=tf.AUTO_REUSE):
                sigma = tf.log(1 + tf.exp(1 + self.dense(h_bar))) # don't need tf.diag() here
        return sigma
    
    def get_decoder_initial_state(self, decoder_cell,
                                  context_final_state=None):
        with tf.variable_scope("decoder", reuse=tf.AUTO_REUSE):            
            decoder_zero_state = tf.cond(
                self.is_training,
                lambda: decoder_cell.zero_state(self.batch_size_per_gpu, tf.float32),
                lambda: decoder_cell.zero_state(
                    self.batch_size_per_gpu * self.beam_width, tf.float32))
            # applies to the first turn since there's no context output (i.e., no source utterance) yet
            if context_final_state is None: 
                decoder_initial_state = decoder_zero_state
            else:
                decoder_initial_state = decoder_zero_state.clone(
                    cell_state=self.tile_if_not_training(context_final_state))
        return decoder_initial_state

    def decode_beam(self, 
                    embedding, decoder_cell, 
                    decoder_initial_state, 
                    output_layer):
        with tf.variable_scope("decoder", reuse=tf.AUTO_REUSE):
            beam_search_decoder = tf.contrib.seq2seq.BeamSearchDecoder(
                decoder_cell, embedding, 
                self.start_tokens, self.end_token,
                decoder_initial_state, self.beam_width,
                output_layer=output_layer,
                length_penalty_weight=self.length_penalty_weight)
            output_beam = tf.contrib.seq2seq.dynamic_decode(
                beam_search_decoder, 
    #             impute_finished=True, # cannot be used with Beamsearch
                maximum_iterations=self.max_iterations, 
                swap_memory=True)         
            sample_ids_beam = output_beam[0].predicted_ids[:, :, 0]
            final_lengths_beam = output_beam[2][:, 0]
        return (sample_ids_beam, final_lengths_beam)
    
    def decode_train(self, target, target_lengths, embedding,
                     decoder_cell, decoder_initial_state, 
                     output_layer):
        with tf.variable_scope("embedding", reuse=tf.AUTO_REUSE):
            embedded_target = tf.nn.embedding_lookup(embedding, target)
        # training helper (for teacher forcing)
        with tf.variable_scope("decoder", reuse=tf.AUTO_REUSE):
            helper_train = tf.contrib.seq2seq.TrainingHelper(
                embedded_target[:, :(-1), :], # get rid of end_token
                target_lengths - 1) # the length is thus decreased by 1
                        
            (decoder_outputs_train, _) = decode(
                decoder_cell, helper_train,
                decoder_initial_state, None,
                output_layer=output_layer) # we don't set max_iter here for training
            
            (logits, _) = decoder_outputs_train

            decoder_mask_float = self.create_decoder_mask(target, target_lengths)
            loss = tf.contrib.seq2seq.sequence_loss(
                logits, target[:, 1:], decoder_mask_float[:, 1:],
                average_across_timesteps=False,
                average_across_batch=False) # get rid of start_token
            total_loss = tf.reduce_sum(loss)
            num_tokens = tf.reduce_sum(decoder_mask_float[:, 1:])    
        
        return (total_loss, num_tokens)
    
    def encode(self,
               cell_fw, cell_bw,
               inputs, sequence_length, 
               embedding,
               reuse=False):
        with tf.variable_scope("embedding", reuse=tf.AUTO_REUSE):
            embedded_inputs = tf.nn.embedding_lookup(embedding, inputs)
            
        with tf.variable_scope("encoder", reuse=tf.AUTO_REUSE):
            (outputs, final_states) = tf.nn.bidirectional_dynamic_rnn(
                cell_fw, cell_bw, embedded_inputs,
                sequence_length=sequence_length,
                dtype=tf.float32, 
                swap_memory=True
            )
                        
            output = tf.concat(outputs, axis=2)
            final_state_h = tf.concat(
                [final_states[0].h, final_states[1].h],
                axis=1)

        return (output, final_state_h)

    def create_placeholders(self):
        self.dialogue = tf.placeholder(
            tf.string, [self.batch_size, self.max_dialogue_length], "dialogue")
        self.turn_length = tf.placeholder(
            tf.int32, [self.batch_size, self.max_dialogue_length], name="utterance_length")
        
        self.keep_prob = tf.placeholder(
            tf.float32, shape=[], name="keep_prob")
        self.is_training = tf.placeholder(
            tf.bool, shape=[], name="is_training")

        self.start_turn_index = tf.placeholder(
            tf.int32, shape=[self.batch_size], name="start_turn_index")

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

