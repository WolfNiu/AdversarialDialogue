
# coding: utf-8

# In[1]:


import tensorflow as tf


# In[ ]:


def double(t):
    return tf.concat([t, t], axis=0)


# In[ ]:


def sequence_loss(logits, sample_ids, final_lengths):
    decoder_mask = tf.sequence_mask(final_lengths, dtype=tf.float32)
    loss = tf.contrib.seq2seq.sequence_loss(
        logits, sample_ids, decoder_mask,
        average_across_timesteps=False,
        average_across_batch=False)
    batch_loss = tf.reduce_sum(loss, axis=1) # only sum along axis 1
    num_tokens = tf.reduce_sum(decoder_mask, axis=1)
    return (batch_loss, num_tokens)


# In[ ]:


def convert_to_number(string_tensor, default_value='0'):
    """
    string_tensor: a 1-D string tensor
    """
    tokenized = tf.sparse_tensor_to_dense(
        tf.string_split(string_tensor, delimiter=' ', skip_empty=True),
        default_value=default_value)
    number = tf.string_to_number(tokenized, out_type=tf.int32)
    return number


# In[2]:


def gpu_config():
    """
    Speicify configurations of GPU
    """
    config = tf.ConfigProto(
        allow_soft_placement=True, log_device_placement=False)
    config.gpu_options.allow_growth = True
    config.gpu_options.allocator_type = 'BFC'
    
    upper = 0.75
    config.gpu_options.per_process_gpu_memory_fraction = upper
    print("GPU memory upper bound:", upper)
    return config


# In[ ]:


def decode(cell, helper, initial_state, max_iterations, output_layer=None):
    decoder = tf.contrib.seq2seq.BasicDecoder(
        cell, helper, initial_state, output_layer=output_layer)
    (decoder_outputs, _, final_lengths) = tf.contrib.seq2seq.dynamic_decode(
        decoder, impute_finished=True,
        maximum_iterations=max_iterations, swap_memory=True)
    return (decoder_outputs, final_lengths)


# In[4]:


def attention(cell, 
              attention_size, attention_layer_size,
              memory, memory_seq_lengths, monotonic=False):
    attention_fn = tf.contrib.seq2seq.BahdanauMonotonicAttention if monotonic else tf.contrib.seq2seq.BahdanauAttention
    
    attention_mechanism = attention_fn(
        attention_size, 
        memory,
        memory_sequence_length=memory_seq_lengths)
    cell_attention = tf.contrib.seq2seq.AttentionWrapper(
        cell, attention_mechanism,
        attention_layer_size=256, # so that context and LSTM output is mixed together
        alignment_history=False, # Set to "False" for beam search!!
        output_attention=False) # behavior of BahdanauAttention
    return cell_attention


# In[5]:


def pad_and_truncate(sample_ids, lengths, max_iterations):
    max_length = tf.reduce_max(lengths)
    padded_sample_ids = tf.pad(
         sample_ids, 
         [[0, 0], 
          [0, max_iterations - max_length]])
    truncated_sample_ids = padded_sample_ids[:, :max_iterations] # truncate length
#     truncated_sample_ids = tf.cond(
#         tf.greater_equal(max_length, max_iterations),
#         lambda: sample_ids[:, :max_iterations],
#         lambda: tf.pad(
#             sample_ids, 
#             [[0, 0], 
#              [0, max_iterations - max_length]]))
    return truncated_sample_ids


# In[6]:


def dropout(cell, keep_prob):
    cell_dropout = tf.contrib.rnn.DropoutWrapper(
        cell,
        output_keep_prob=keep_prob,
        variational_recurrent=True, dtype=tf.float32)        
    return cell_dropout


# In[7]:


def create_cell(hidden_size, keep_prob, 
                num_proj=None,
                attention_size=512, attention_layer_size=256,
                memory=None, memory_seq_lengths=None, 
                reuse=False, monotonic=False):
    cell = tf.contrib.rnn.LSTMCell(
        hidden_size, use_peepholes=True, # peephole: allow implementation of LSTMP
        initializer=tf.contrib.layers.xavier_initializer(),
        forget_bias=1.0, reuse=reuse)
    cell = dropout(cell, keep_prob)
    # Note that the attention wrapper HAS TO come before projection wrapper,
    # Otherwise the attention weights will not work correctly.
    if (memory is not None) and (memory_seq_lengths is not None):
        cell = attention(
            cell, 
            attention_size, attention_layer_size,
            memory, memory_seq_lengths,
            monotonic=monotonic)
    if num_proj is not None:
        cell = tf.contrib.rnn.OutputProjectionWrapper(cell, num_proj)
    return cell


# In[8]:


def create_MultiRNNCell(hidden_sizes, keep_prob, 
                        num_proj=None,
                        attention_size=512, attention_layer_size=256,
                        memory=None, memory_seq_lengths=None, 
                        reuse=False, monotonic=False):
    """
    Only the last layer has projection and attention
    Args:
        hidden_sizes: a list of hidden sizes for each layer
        num_proj: the projection size
    Returns:
        A cell or a wrapped rnn cell
    """
    assert len(hidden_sizes) > 0

    if len(hidden_sizes) == 1:
        cell_first = create_cell(
            hidden_sizes[0], 
            keep_prob, 
            num_proj=num_proj, 
            memory=memory, memory_seq_lengths=memory_seq_lengths, 
            reuse=reuse, monotonic=monotonic)
        return cell_first
    else: # if there are at least two layers        
        cell_first = create_cell(
            hidden_sizes[0], 
            keep_prob, 
            num_proj=None, 
            memory=None, memory_seq_lengths=None, 
            reuse=reuse, monotonic=monotonic)
        cell_last = create_cell(
            hidden_sizes[-1],
            keep_prob,
            num_proj=num_proj,
            memory=memory, memory_seq_lengths=memory_seq_lengths, 
            reuse=reuse, monotonic=monotonic)
        cells_in_between = [
            create_cell(
                hidden_size, 
                keep_prob, 
                num_proj=None, 
                memory=None, memory_seq_lengths=None, 
                reuse=reuse, monotonic=monotonic)
            for hidden_size
            in hidden_sizes[1:(-1)]]
        return tf.contrib.rnn.MultiRNNCell(
            [cell_first] + cells_in_between + [cell_last])


# In[9]:


def bidirecitonal_dynamic_lstm(cell_fw, cell_bw,
                               inputs, seq_lengths):
    (outputs, final_states) = tf.nn.bidirectional_dynamic_rnn(
        cell_fw, cell_bw, inputs,                    
        sequence_length=seq_lengths,
        dtype=tf.float32,
        swap_memory=True)
    output = tf.concat(outputs, axis=2)

    (final_states_fw, final_states_bw) = final_states
#     if num_layers_decoder == 1:
#         final_state_fw_c = final_states_fw[0]
#         final_state_fw_h = final_states_fw[1]
#         final_state_bw_c = final_states_bw[0]
#         final_state_bw_h = final_states_bw[1]        
#     else:
    final_states = [
        tf.contrib.rnn.LSTMStateTuple(
            tf.concat(
                [final_state_fw.c, final_state_bw.c], 
                axis=1),
            tf.concat(
                [final_state_fw.h, final_state_bw.h], 
                axis=1))
        for (final_state_fw, final_state_bw)
        in zip(final_states_fw, final_states_bw)]
    return (output, tuple(final_states))


# In[10]:


def get_mask(seqs, indices):
    tensor = tf.convert_to_tensor(indices)
    bool_matrix = tf.equal(
        tf.expand_dims(seqs, axis=0),
        tf.reshape(tensor, [len(indices), 1, 1]))
    mask = tf.reduce_any(bool_matrix, axis=0)
    return mask


# In[11]:


def average_gradients(tower_grads):
    """
    Copied from: https://github.com/tensorflow/tensorflow/blob/r0.7/tensorflow/models/image/cifar10/cifar10_multi_gpu_train.py

    Calculate the average gradient for each shared variable across all towers.
    Note that this function provides a synchronization point across all towers.
    Args:
        tower_grads: List of lists of (gradient, variable) tuples. 
            The outer list is over individual gradients. 
            The inner list is over the gradient calculation for each tower.
    Returns:
        List of pairs of (gradient, variable) where the gradient has been 
        averaged across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(grads, 0)
        grad = tf.reduce_mean(grad, axis=0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1] # [0]: first tower [1]: ref to var
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


# In[12]:


def compute_grads(loss, optimizer, var_list=None):
    grads = optimizer.compute_gradients(loss, var_list=var_list)
    valid_grads = [
        (grad, var) 
        for (grad, var) in grads 
        if grad is not None]
    if len(valid_grads) != len(var_list):
        print("Warning: some grads are None.")
    return valid_grads


# In[13]:


"""
Compute average gradients, perform gradient clipping and apply gradients
Args:
    tower_grad: gradients collected from all GPUs
Returns:
    the op of apply_gradients
"""

def apply_grads(optimizer, tower_grads, clipping_threshold=5.0, global_step=None):
    # averaging over all gradients
    avg_grads = average_gradients(tower_grads)

    # Perform gradient clipping
    (gradients, variables) = zip(*avg_grads)
    (clipped_gradients, _) = tf.clip_by_global_norm(gradients, clipping_threshold)

    # Apply the gradients to adjust the shared variables.
    apply_gradients_op = optimizer.apply_gradients(
        zip(clipped_gradients, variables), global_step=global_step)

    return apply_gradients_op


# In[14]:


def tile_single_cell_state(state, beam_width):
    """
    Takes in a cell state and return its tiled version
    """
    if isinstance(state, tf.Tensor):
        s = tf.contrib.seq2seq.tile_batch(state, beam_width)
        if s is None:
            print("Got it!")
            print(state)
        return s
    elif isinstance(state, tf.contrib.rnn.LSTMStateTuple):
        return tf.contrib.rnn.LSTMStateTuple(
            tile_single_cell_state(state.c, beam_width), 
            tile_single_cell_state(state.h, beam_width))
    elif isinstance(state, tf.contrib.seq2seq.AttentionWrapperState):
        return tf.contrib.seq2seq.AttentionWrapperState(
            tile_single_cell_state(state.cell_state, beam_width), 
            tile_single_cell_state(state.attention, beam_width), 
            state.time, tile_single_cell_state(state.alignments, beam_width), 
            state.alignment_history)
    return None


# In[15]:


def tile_multi_cell_state(states, beam_width):
    return tuple([tile_single_cell_state(state, beam_width) for state in states])

