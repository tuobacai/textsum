#coding:utf-8
import tensorflow as tf
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops

class Seq2SeqModel(object):
    def __init__(self,
                 buckets,
                 size,
                 from_vocab_size,
                 target_vocab_size,
                 num_layers,
                 max_gradient_norm,
                 batch_size,
                 learning_rate,
                 learning_rate_decay_factor,
                 withAdagrad=True,
                 forward_only=False,
                 dropoutRate=1.0,
                 devices="",
                 run_options=None,
                 run_metadata=None,
                 topk_n=30,
                 dtype=tf.float32,
                 with_attention=False,
                 beam_search=False,
                 beam_buckets=None
                 ):
        """Create the model.

        Args:
        buckets: a list of pairs (I, O), where I specifies maximum input length
        that will be processed in that bucket, and O specifies maximum output
        length. Training instances that have inputs longer than I or outputs
        longer than O will be pushed to the next bucket and padded accordingly.
        We assume that the list is sorted, e.g., [(2, 4), (8, 16)].
        size: number of units in each layer of the model.
        num_layers: number of layers in the model.
        max_gradient_norm: gradients will be clipped to maximally this norm.
        batch_size: the size of the batches used during training;
        the model construction is independent of batch_size, so it can be
        changed after initialization if this is convenient, e.g., for decoding.

        learning_rate: learning rate to start with.
        learning_rate_decay_factor: decay learning rate by this much when needed.

        forward_only: if set, we do not construct the backward pass in the model.
        dtype: the data type to use to store internal variables.
        """
        self.buckets = buckets
        self.PAD_ID = 0
        self.GO_ID = 1
        self.EOS_ID = 2
        self.UNK_ID = 3
        self.batch_size = batch_size
        self.devices = devices
        self.run_options = run_options
        self.run_metadata = run_metadata
        self.topk_n = topk_n
        self.dtype = dtype
        self.from_vocab_size = from_vocab_size
        self.target_vocab_size = target_vocab_size
        self.num_layers = num_layers
        self.size = size
        self.with_attention = with_attention
        self.beam_search = beam_search

        # some parameters
        with tf.device(devices[0]):
            self.dropoutRate = tf.Variable(
                float(dropoutRate), trainable=False, dtype=dtype)
            self.dropoutAssign_op = self.dropoutRate.assign(dropoutRate)
            self.dropout10_op = self.dropoutRate.assign(1.0)
            self.learning_rate = tf.Variable(
                float(learning_rate), trainable=False, dtype=dtype)
            self.learning_rate_decay_op = self.learning_rate.assign(
                self.learning_rate * learning_rate_decay_factor)
            self.global_step = tf.Variable(0, trainable=False)

        # Input Layer
        with tf.device(devices[0]):
            # for encoder
            self.sources = []
            self.sources_embed = []

            self.source_input_embedding = tf.get_variable("source_input_embedding", [from_vocab_size, size],
                                                          dtype=dtype)

            for i in xrange(buckets[-1][0]):
                source_input_plhd = tf.placeholder(tf.int32, shape=[self.batch_size], name="source{}".format(i))
                source_input_embed = tf.nn.embedding_lookup(self.source_input_embedding, source_input_plhd)
                self.sources.append(source_input_plhd)
                self.sources_embed.append(source_input_embed)

            # for decoder
            self.inputs = []
            self.inputs_embed = []

            self.input_embedding = tf.get_variable("input_embedding", [target_vocab_size, size], dtype=dtype)

            for i in xrange(buckets[-1][1]):
                input_plhd = tf.placeholder(tf.int32, shape=[self.batch_size], name="input{}".format(i))
                input_embed = tf.nn.embedding_lookup(self.input_embedding, input_plhd)
                self.inputs.append(input_plhd)
                self.inputs_embed.append(input_embed)

        def lstm_cell():
            cell = tf.contrib.rnn.LSTMCell(size, state_is_tuple=True)
            cell = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=self.dropoutRate)
            return cell

        # LSTM
        with tf.device(devices[1]):
            # for encoder
            if num_layers == 1:
                encoder_cell = lstm_cell()
            else:
                encoder_cell = tf.contrib.rnn.MultiRNNCell([lstm_cell() for _ in xrange(num_layers)],
                                                           state_is_tuple=True)
            encoder_cell = tf.contrib.rnn.DropoutWrapper(encoder_cell, output_keep_prob=self.dropoutRate)

            # for decoder
            if num_layers == 1:
                decoder_cell = lstm_cell()
            else:
                decoder_cell = tf.contrib.rnn.MultiRNNCell([lstm_cell() for _ in xrange(num_layers)],
                                                           state_is_tuple=True)
            decoder_cell = tf.contrib.rnn.DropoutWrapper(decoder_cell, output_keep_prob=self.dropoutRate)

        self.encoder_cell = encoder_cell
        self.decoder_cell = decoder_cell

        # Output Layer
        with tf.device(devices[2]):
            self.targets = []
            self.target_weights = []

            self.output_embedding = tf.get_variable("output_embedding", [target_vocab_size, size], dtype=dtype)
            self.output_bias = tf.get_variable("output_bias", [target_vocab_size], dtype=dtype)

            # target: 1  2  3  4
            # inputs: go 1  2  3
            # weights:1  1  1  1

            for i in xrange(buckets[-1][1]):
                self.targets.append(tf.placeholder(tf.int32,
                                                   shape=[self.batch_size], name="target{}".format(i)))
                self.target_weights.append(tf.placeholder(dtype,
                                                          shape=[self.batch_size], name="target_weight{}".format(i)))

        if not beam_search:
            # Model with buckets
            self.model_with_buckets(self.sources_embed, self.inputs_embed, self.targets, self.target_weights,
                                    self.buckets, encoder_cell, decoder_cell, dtype, devices=devices,
                                    attention=with_attention)

            # train
            with tf.device(devices[0]):
                params = tf.trainable_variables()
                if not forward_only:
                    self.gradient_norms = []
                    self.updates = []
                    if withAdagrad:
                        opt = tf.train.AdagradOptimizer(self.learning_rate)
                    else:
                        opt = tf.train.GradientDescentOptimizer(self.learning_rate)

                    for b in xrange(len(buckets)):
                        gradients = tf.gradients(self.losses[b], params, colocate_gradients_with_ops=True)
                        clipped_gradients, norm = tf.clip_by_global_norm(gradients, max_gradient_norm)
                        self.gradient_norms.append(norm)
                        self.updates.append(
                            opt.apply_gradients(zip(clipped_gradients, params), global_step=self.global_step))


        else:  # for beam search

            self.init_beam_decoder(beam_buckets)

        all_vars = tf.global_variables()
        new_all_vars = []
        for var in all_vars:
            if not var.name.startswith("beam_search"):
                new_all_vars.append(var)

        self.saver = tf.train.Saver(new_all_vars)
        self.best_saver = tf.train.Saver(new_all_vars)

    ######### Train ##########

    def step(self, session, sources, inputs, targets, target_weights,
             bucket_id, forward_only=False, dump_lstm=False):

        source_length, target_length = self.buckets[bucket_id]

        input_feed = {}

        for l in xrange(source_length):
            input_feed[self.sources[l].name] = sources[l]

        for l in xrange(target_length):
            input_feed[self.inputs[l].name] = inputs[l]
            input_feed[self.targets[l].name] = targets[l]
            input_feed[self.target_weights[l].name] = target_weights[l]

        # output_feed
        if forward_only:
            output_feed = [self.losses[bucket_id]]
        else:
            output_feed = [self.losses[bucket_id]]
            output_feed += [self.updates[bucket_id], self.gradient_norms[bucket_id]]

        outputs = session.run(output_feed, input_feed, options=self.run_options, run_metadata=self.run_metadata)

        if forward_only and dump_lstm:
            return outputs
        else:
            return outputs[0]  # only return losses




    def model_with_buckets(self, sources, inputs, targets, weights,
                       buckets, encoder_cell, decoder_cell, dtype,
                       per_example_loss=False, name=None, devices = None, attention = False):

        losses = []
        hts = []
        logits = []
        topk_values = []
        topk_indexes = []

        seq2seq_f = None

        if attention:
            seq2seq_f = None
        else:
            seq2seq_f = self.basic_seq2seq

        # softmax
        with tf.device(devices[2]):
            softmax_loss_function = lambda x,y: tf.nn.sparse_softmax_cross_entropy_with_logits(logits=x, labels= y)



        for j, (source_length, target_length) in enumerate(buckets):
            with variable_scope.variable_scope(variable_scope.get_variable_scope(),reuse=True if j > 0 else None):

                _hts, decoder_state = seq2seq_f(encoder_cell, decoder_cell, sources[:source_length], inputs[:target_length], dtype, devices)

                hts.append(_hts)

                # logits / loss / topk_values + topk_indexes
                with tf.device(devices[2]):
                    _logits = [ tf.add(tf.matmul(ht, tf.transpose(self.output_embedding)), self.output_bias) for ht in _hts]
                    logits.append(_logits)

                    if per_example_loss:
                        losses.append(sequence_loss_by_example(
                                logits[-1], targets[:target_length], weights[:target_length],
                                softmax_loss_function=softmax_loss_function))

                    else:
                        losses.append(sequence_loss(
                                logits[-1], targets[:target_length], weights[:target_length],
                                softmax_loss_function=softmax_loss_function))

                    topk_value, topk_index = [], []

                    for _logits in logits[-1]:
                        value, index = tf.nn.top_k(tf.nn.softmax(_logits), self.topk_n, sorted = True)
                        topk_value.append(value)
                        topk_index.append(index)
                    topk_values.append(topk_value)
                    topk_indexes.append(topk_index)

        self.losses = losses
        self.hts = hts
        self.logits = logits
        self.topk_values = topk_values
        self.topk_indexes = topk_indexes

    #基础的seq2seq
    def basic_seq2seq(self, encoder_cell, decoder_cell, encoder_inputs, decoder_inputs, dtype, devices=None):

        # initial state
        with tf.variable_scope("basic_seq2seq"):
            with tf.device(devices[1]):
                init_state = encoder_cell.zero_state(self.batch_size, dtype)

                with tf.variable_scope("encoder"):
                    encoder_outputs, encoder_state = tf.contrib.rnn.static_rnn(encoder_cell, encoder_inputs,
                                                                               initial_state=init_state)

                with tf.variable_scope("decoder"):
                    decoder_outputs, decoder_state = tf.contrib.rnn.static_rnn(decoder_cell, decoder_inputs,
                                                                               initial_state=encoder_state)

        return decoder_outputs, decoder_state



    ######### Beam Search ##########

    def init_beam_decoder(self, beam_buckets):

        self.beam_buckets = beam_buckets

        # before and after state

        self.before_state = []
        self.after_state = []

        shape = [self.batch_size, self.size]

        with tf.device(self.devices[0]):
            with tf.variable_scope("beam_search"):
                # place_holders

                # self.source_length = tf.placeholder(tf.int32, shape=[1], name = "source_length")
                self.beam_parent = tf.placeholder(tf.int32, shape=[self.batch_size], name="beam_parent")

                self.zero_beam_parent = [0] * self.batch_size

                # two variable: before_state, after_state
                for i in xrange(self.num_layers):
                    cb = tf.get_variable("before_c_{}".format(i), shape, initializer=tf.constant_initializer(0.0),
                                         trainable=False)
                    hb = tf.get_variable("before_h_{}".format(i), shape, initializer=tf.constant_initializer(0.0),
                                         trainable=False)
                    sb = tf.contrib.rnn.LSTMStateTuple(cb, hb)
                    ca = tf.get_variable("after_c_{}".format(i), shape, initializer=tf.constant_initializer(0.0),
                                         trainable=False)
                    ha = tf.get_variable("after_h_{}".format(i), shape, initializer=tf.constant_initializer(0.0),
                                         trainable=False)
                    sa = tf.contrib.rnn.LSTMStateTuple(ca, ha)
                    self.before_state.append(sb)
                    self.after_state.append(sa)


                    # after2before_ops
            self.after2before_ops = self.after2before(self.beam_parent)

            # encoder and one-step decoder
            self.beam_with_buckets(self.sources_embed, self.inputs_embed, self.beam_buckets, self.encoder_cell,
                                   self.decoder_cell, self.dtype, self.devices, self.with_attention)

    def after2before(self, beam_parent):
        # beam_parent : [beam_size]
        ops = []
        for i in xrange(len(self.after_state)):
            c = self.after_state[i].c
            h = self.after_state[i].h
            new_c = tf.nn.embedding_lookup(c, beam_parent)
            new_h = tf.nn.embedding_lookup(h, beam_parent)
            copy_c = self.before_state[i].c.assign(new_c)
            copy_h = self.before_state[i].h.assign(new_h)
            ops.append(copy_c)
            ops.append(copy_h)

        return ops

    def beam_with_buckets(self, sources, inputs, source_buckets, encoder_cell, decoder_cell, dtype, devices=None,
                          attention=False):

        self.hts = []
        self.topk_values = []
        self.eos_values = []
        self.topk_indexes = []

        self.encoder2before_ops = []
        self.decoder2after_ops = []

        for j, source_length in enumerate(source_buckets):
            with variable_scope.variable_scope(variable_scope.get_variable_scope(), reuse=True if j > 0 else None):

                # seq2seq
                if not attention:
                    _hts, _, e2b, d2a = self.beam_basic_seq2seq(encoder_cell, decoder_cell, sources[:source_length],
                                                                inputs[:1], dtype, devices)
                    self.hts.append(_hts)
                    self.encoder2before_ops.append(e2b)
                    self.decoder2after_ops.append(d2a)
                else:
                    pass

                # logits
                _softmaxs = [tf.nn.softmax(tf.add(tf.matmul(ht, tf.transpose(self.output_embedding)), self.output_bias))
                             for ht in _hts]

                # topk
                topk_value, topk_index, eos_value = [], [], []

                for _softmax in _softmaxs:
                    value, index = tf.nn.top_k(_softmax, self.topk_n, sorted=True)
                    eos_v = tf.slice(_softmax, [0, self.EOS_ID], [-1, 1])

                    topk_value.append(value)
                    topk_index.append(index)
                    eos_value.append(eos_v)

                self.topk_values.append(topk_value)
                self.topk_indexes.append(topk_index)
                self.eos_values.append(eos_value)

    def beam_basic_seq2seq(self, encoder_cell, decoder_cell, encoder_inputs, decoder_inputs, dtype,
                           devices=None):
        scope_name = "basic_seq2seq"
        with tf.variable_scope(scope_name):
            init_state = encoder_cell.zero_state(self.batch_size, dtype)

            with tf.variable_scope("encoder"):
                encoder_outputs, encoder_state = tf.contrib.rnn.static_rnn(encoder_cell, encoder_inputs,
                                                                           initial_state=init_state)

            # encoder -> before state
            encoder2before_ops = self.states2states(encoder_state, self.before_state)

            with tf.variable_scope("decoder"):
                # One step encoder: starts from before_state
                decoder_outputs, decoder_state = tf.contrib.rnn.static_rnn(decoder_cell, decoder_inputs,
                                                                           initial_state=self.before_state)

            # decoder_state -> after state
            decoder2after_ops = self.states2states(decoder_state, self.after_state)

        return decoder_outputs, decoder_state, encoder2before_ops, decoder2after_ops

    def states2states(self, states, to_states):
        ops = []
        for i in xrange(len(states)):
            copy_c = to_states[i].c.assign(states[i].c)
            copy_h = to_states[i].h.assign(states[i].h)
            ops.append(copy_c)
            ops.append(copy_h)

        return ops

    def beam_step(self, session, bucket_id, index=0, sources=None, target_inputs=None, beam_parent=None):

        if index == 0:
            # go through the source by LSTM
            input_feed = {}
            for i in xrange(len(sources)):
                input_feed[self.sources[i].name] = sources[i]

            output_feed = []
            output_feed += self.encoder2before_ops[bucket_id]
            _ = session.run(output_feed, input_feed)

        else:
            # copy the after_state to before states

            input_feed = {}
            input_feed[self.beam_parent.name] = beam_parent
            output_feed = []
            output_feed.append(self.after2before_ops)
            _ = session.run(output_feed, input_feed)

        # Run one step of RNN

        input_feed = {}

        input_feed[self.inputs[0].name] = target_inputs  # [batch_size]

        output_feed = {}
        output_feed['value'] = self.topk_values[bucket_id]
        output_feed['index'] = self.topk_indexes[bucket_id]
        output_feed['eos_value'] = self.eos_values[bucket_id]
        output_feed['ops'] = self.decoder2after_ops[bucket_id]

        outputs = session.run(output_feed, input_feed)

        return outputs['value'], outputs['index'], outputs['eos_value']


            ############ loss function ###########

def sequence_loss_by_example(logits, targets, weights,
                             average_across_timesteps=True,
                             softmax_loss_function=None, name=None):
  """Weighted cross-entropy loss for a sequence of logits (per example).

  Args:
    logits: List of 2D Tensors of shape [batch_size x num_decoder_symbols].
    targets: List of 1D batch-sized int32 Tensors of the same length as logits.
    weights: List of 1D batch-sized float-Tensors of the same length as logits.
    average_across_timesteps: If set, divide the returned cost by the total
      label weight.
    softmax_loss_function: Function (inputs-batch, labels-batch) -> loss-batch
      to be used instead of the standard softmax (the default if this is None).
    name: Optional name for this operation, default: "sequence_loss_by_example".

  Returns:
    1D batch-sized float Tensor: The log-perplexity for each sequence.

  Raises:
    ValueError: If len(logits) is different from len(targets) or len(weights).
  """
  if len(targets) != len(logits) or len(weights) != len(logits):
    raise ValueError("Lengths of logits, weights, and targets must be the same "
                     "%d, %d, %d." % (len(logits), len(weights), len(targets)))
  with tf.name_scope(name, "sequence_loss_by_example", logits + targets + weights):
  # with ops.op_scope(logits + targets + weights,name, "sequence_loss_by_example"):
    log_perp_list = []
    for logit, target, weight in zip(logits, targets, weights):
      if softmax_loss_function is None:
        # TODO(irving,ebrevdo): This reshape is needed because
        # sequence_loss_by_example is called with scalars sometimes, which
        # violates our general scalar strictness policy.
        target = array_ops.reshape(target, [-1])
        crossent = nn_ops.sparse_softmax_cross_entropy_with_logits(
            logit, target)
      else:
        crossent = softmax_loss_function(logit, target)
      log_perp_list.append(crossent * weight)

    log_perps = math_ops.add_n(log_perp_list)
    if average_across_timesteps:
      total_size = math_ops.add_n(weights)
      total_size += 1e-12  # Just to avoid division by 0 for all-0 weights.
      log_perps /= total_size
  return log_perps

def sequence_loss(logits, targets, weights,
                average_across_timesteps=False, average_across_batch=False,
                softmax_loss_function=None, name=None):
    """Weighted cross-entropy loss for a sequence of logits, batch-collapsed.

    Args:
      logits: List of 2D Tensors of shape [batch_size x num_decoder_symbols].
      targets: List of 1D batch-sized int32 Tensors of the same length as logits.
      weights: List of 1D batch-sized float-Tensors of the same length as logits.
      average_across_timesteps: If set, divide the returned cost by the total
        label weight.
      average_across_batch: If set, divide the returned cost by the batch size.
      softmax_loss_function: Function (inputs-batch, labels-batch) -> loss-batch
        to be used instead of the standard softmax (the default if this is None).
      name: Optional name for this operation, defaults to "sequence_loss".

    Returns:
      A scalar float Tensor: The average log-perplexity per symbol (weighted).

    Raises:
      ValueError: If len(logits) is different from len(targets) or len(weights).
    """

    with tf.name_scope(name, "sequence_loss", logits + targets + weights):
        # with ops.op_scope(logits + targets + weights, name, "sequence_loss"):
        cost = math_ops.reduce_sum(sequence_loss_by_example(
            logits, targets, weights,
            average_across_timesteps=average_across_timesteps,
            softmax_loss_function=softmax_loss_function))
        if average_across_batch:
            total_size = tf.reduce_sum(tf.sign(weights[0]))
            return cost / math_ops.cast(total_size, cost.dtype)
        else:
            return cost

