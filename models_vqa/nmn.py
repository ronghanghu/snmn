import numpy as np
import tensorflow as tf
from tensorflow import convert_to_tensor as to_T, newaxis as ax

from .config import cfg
from util.cnn import fc_layer as fc, conv_layer as conv

MODULE_INPUT_NUM = {
    '_NoOp': 0,
    '_Find': 0,
    '_Transform': 1,
    '_Filter': 1,
    '_And': 2,
    '_Describe': 1,
}

MODULE_OUTPUT_NUM = {
    '_NoOp': 0,
    '_Find': 1,
    '_Transform': 1,
    '_Filter': 1,
    '_And': 1,
    '_Describe': 1,
}


class NMN:
    def __init__(self, kb_batch, c_list, module_names, module_prob_list,
                 scope='NMN', reuse=None):
        """
        NMN v4 with an attention stack
        """
        with tf.variable_scope(scope, reuse=reuse):
            self.kb_batch = kb_batch
            self.c_list = c_list
            self.module_prob_list = module_prob_list

            self.T_ctrl = cfg.MODEL.T_CTRL
            self.mem_dim = cfg.MODEL.NMN.MEM_DIM
            self.N = tf.shape(kb_batch)[0]
            self.H = tf.shape(kb_batch)[1]
            self.W = tf.shape(kb_batch)[2]
            self.att_shape = to_T([self.N, self.H, self.W, 1])

            self.stack_len = cfg.MODEL.NMN.STACK.LENGTH
            # The initialial stack values are all zeros everywhere
            self.att_stack_init = tf.zeros(
                to_T([self.N, self.H, self.W, self.stack_len]))
            # The initial stack pointer points to the stack bottom
            self.stack_ptr_init = tf.one_hot(
                tf.zeros(to_T([self.N]), tf.int32), self.stack_len)
            self.mem_init = tf.zeros(to_T([self.N, self.mem_dim]))

            # zero-outputs that can be easily used by the modules
            self.att_zero = tf.zeros(self.att_shape, tf.float32)
            self.mem_zero = tf.zeros(to_T([self.N, self.mem_dim]), tf.float32)

            # the set of modules and functions (e.g. "_Find" -> Find)
            self.module_names = module_names
            self.module_funcs = [getattr(self, m[1:]) for m in module_names]
            self.module_validity_mat = _build_module_validity_mat(module_names)

            # unroll the modules with a fixed number of timestep T_ctrl
            self.att_stack_list = []
            self.stack_ptr_list = []
            self.mem_list = []
            att_stack_prev = self.att_stack_init
            stack_ptr_prev = self.stack_ptr_init
            mem_prev = self.mem_init
            for t in range(self.T_ctrl):
                c_i = self.c_list[t]
                module_prob = self.module_prob_list[t]
                # only keep the prob of valid modules (i.e. those won't cause
                # stack underflow or overflow. e.g. _Filter can't be run at
                # t = 0 since the stack is empty).
                if cfg.MODEL.NMN.VALIDATE_MODULES:
                    module_validity = tf.matmul(
                        stack_ptr_prev, self.module_validity_mat)
                    if cfg.MODEL.NMN.HARD_MODULE_VALIDATION:
                        module_validity = tf.round(module_validity)
                    module_prob *= module_validity
                    module_prob /= tf.reduce_sum(
                        module_prob, axis=1, keepdims=True)
                    self.module_prob_list[t] = module_prob

                # run all the modules, and average their results wrt module_w
                res = [f(att_stack_prev, stack_ptr_prev, mem_prev, c_i,
                       reuse=(t > 0)) for f in self.module_funcs]

                att_stack_avg = tf.reduce_sum(
                    module_prob[:, ax, ax, ax, :] *
                    tf.stack([r[0] for r in res], axis=4), axis=-1)
                # print and check the attention values
                # att_stack_avg = tf.Print(
                #     att_stack_avg,
                #     [tf.reduce_max(tf.abs(r[0])) for r in res],
                #     message='t = %d, att: ' % t)
                stack_ptr_avg = _sharpen_ptr(tf.reduce_sum(
                    module_prob[:, ax, :] *
                    tf.stack([r[1] for r in res], axis=2), axis=-1))
                mem_avg = tf.reduce_sum(
                    module_prob[:, ax, :] *
                    tf.stack([r[2] for r in res], axis=2), axis=-1)

                self.att_stack_list.append(att_stack_avg)
                self.stack_ptr_list.append(stack_ptr_avg)
                self.mem_list.append(mem_avg)
                att_stack_prev = att_stack_avg
                stack_ptr_prev = stack_ptr_avg
                mem_prev = mem_avg

            self.att_last = _read_from_stack(
                self.att_stack_list[-1], self.stack_ptr_list[-1])
            self.mem_last = self.mem_list[-1]

    def NoOp(self, att_stack, stack_ptr, mem_in, c_i, scope='NoOp',
             reuse=None):
        """
        Does nothing. It leaves the stack pointer, the stack and mem vector
        as-is.
        """
        return att_stack, stack_ptr, mem_in

    def Find(self, att_stack, stack_ptr, mem_in, c_i, scope='Find',
             reuse=None):
        """
        Performs localization, and updates memory vector.
        """
        with tf.variable_scope(scope, reuse=reuse):
            # Get attention
            #   1) linearly map the controller vectors to the KB dimension
            #   2) elementwise product with KB
            #   3) 1x1 convolution to get attention logits
            c_mapped = fc('fc_c_mapped', c_i, output_dim=cfg.MODEL.KB_DIM)
            elt_prod = tf.nn.l2_normalize(
                self.kb_batch * c_mapped[:, ax, ax, :], axis=-1)
            att_out = _1x1conv('conv_att_out', elt_prod, output_dim=1)

            # Push to stack
            stack_ptr = _move_ptr_fw(stack_ptr)
            att_stack = _write_to_stack(att_stack, stack_ptr, att_out)

        return att_stack, stack_ptr, self.mem_zero

    def Transform(self, att_stack, stack_ptr, mem_in, c_i, scope='Transform',
                  reuse=None):
        """
        Transforms the previous attention, and updates memory vector.
        """
        with tf.variable_scope(scope, reuse=reuse):
            # Get attention
            #   1) linearly map the controller vectors to the KB dimension
            #   2) extract attended features from the input attention
            #   2) elementwise product with KB
            #   3) 1x1 convolution to get attention logits

            # Pop from stack
            att_in = _read_from_stack(att_stack, stack_ptr)
            # stack_ptr = _move_ptr_bw(stack_ptr)  # cancel-out below

            c_mapped = fc('fc_c_mapped', c_i, output_dim=cfg.MODEL.KB_DIM)
            kb_att_in = _extract_softmax_avg(self.kb_batch, att_in)
            elt_prod = tf.nn.l2_normalize(
                self.kb_batch * c_mapped[:, ax, ax, :] *
                kb_att_in[:, ax, ax, :], axis=-1)
            att_out = _1x1conv('conv_att_out', elt_prod, output_dim=1)

            # Push to stack
            # stack_ptr = _move_ptr_fw(stack_ptr)  # cancel-out above
            att_stack = _write_to_stack(att_stack, stack_ptr, att_out)

        return att_stack, stack_ptr, self.mem_zero

    def Filter(self, att_stack, stack_ptr, mem_in, c_i, scope='Filter',
               reuse=None):
        """
        Combo of Find + And. First run Find, and then run And.
        """
        # Run Find module
        att_stack, stack_ptr, _ = self.Find(
            att_stack, stack_ptr, mem_in, c_i, reuse=True)
        # Run And module
        att_stack, stack_ptr, _ = self.And(
            att_stack, stack_ptr, mem_in, c_i, reuse=True)

        return att_stack, stack_ptr, self.mem_zero

    def And(self, att_stack, stack_ptr, mem_in, c_i, scope='And', reuse=None):
        """
        Take the intersection between two attention maps
        """
        with tf.variable_scope(scope, reuse=reuse):
            # Get attention
            #   1) Just take the elementwise minimum of the two inputs

            # Pop from stack
            att_in_2 = _read_from_stack(att_stack, stack_ptr)
            stack_ptr = _move_ptr_bw(stack_ptr)
            att_in_1 = _read_from_stack(att_stack, stack_ptr)
            # stack_ptr = _move_ptr_bw(stack_ptr)  # cancel-out below

            att_out = tf.minimum(att_in_1, att_in_2)

            # Push to stack
            # stack_ptr = _move_ptr_fw(stack_ptr)  # cancel-out above
            att_stack = _write_to_stack(att_stack, stack_ptr, att_out)

        return att_stack, stack_ptr, self.mem_zero

    def Describe(self, att_stack, stack_ptr, mem_in, c_i, scope='Describe',
                 reuse=None):
        """
        Describe using one input attention. Outputs zero attention.
        """
        with tf.variable_scope(scope, reuse=reuse):
            # Update memory:
            #   1) linearly map the controller vectors to the KB dimension
            #   2) extract attended features from the input attention
            #   3) elementwise multplication
            #   2) linearly merge with previous memory vector, find memory
            #      vector and control state

            att_stack_old, stack_ptr_old = att_stack, stack_ptr  # make a copy
            # Pop from stack
            att_in = _read_from_stack(att_stack, stack_ptr)
            # stack_ptr = _move_ptr_bw(stack_ptr)  # cancel-out below

            c_mapped = fc('fc_c_mapped', c_i, output_dim=cfg.MODEL.KB_DIM)
            kb_att_in = _extract_softmax_avg(self.kb_batch, att_in)
            elt_prod = tf.nn.l2_normalize(c_mapped * kb_att_in, axis=-1)
            mem_out = fc(
                'fc_mem_out', tf.concat([c_i, mem_in, elt_prod], axis=1),
                output_dim=self.mem_dim)

            # Push to stack
            # stack_ptr = _move_ptr_fw(stack_ptr)  # cancel-out above
            att_stack = _write_to_stack(att_stack, stack_ptr, self.att_zero)

            if cfg.MODEL.NMN.DESCRIBE_ONE.KEEP_STACK:
                att_stack, stack_ptr = att_stack_old, stack_ptr_old

        return att_stack, stack_ptr, mem_out


def _move_ptr_fw(stack_ptr):
    """
    Move the stack pointer forward (i.e. to push to stack).
    """
    # Note: in TF, conv1d is implemented as auto-correlation (instead of
    # mathmatical convolution), so no flipping of the filter.
    filter_fw = to_T(np.array([1, 0, 0], np.float32).reshape((3, 1, 1)))
    new_stack_ptr = tf.squeeze(
        tf.nn.conv1d(stack_ptr[..., ax], filter_fw, 1, 'SAME'), axis=[2])
    # when the stack pointer is already at the stack top, keep
    # the pointer in the same location (otherwise the pointer will be all zero)
    if cfg.MODEL.NMN.STACK.GUARD_STACK_PTR:
        stack_len = cfg.MODEL.NMN.STACK.LENGTH
        stack_top_mask = tf.one_hot(stack_len - 1, stack_len)
        new_stack_ptr += stack_top_mask * stack_ptr
    return new_stack_ptr


def _move_ptr_bw(stack_ptr):
    """
    Move the stack pointer backward (i.e. to pop from stack).
    """
    # Note: in TF, conv1d is implemented as auto-correlation (instead of
    # mathmatical convolution), so no flipping of the filter.
    filter_fw = to_T(np.array([0, 0, 1], np.float32).reshape((3, 1, 1)))
    new_stack_ptr = tf.squeeze(
        tf.nn.conv1d(stack_ptr[..., ax], filter_fw, 1, 'SAME'), axis=[2])
    # when the stack pointer is already at the stack bottom, keep
    # the pointer in the same location (otherwise the pointer will be all zero)
    if cfg.MODEL.NMN.STACK.GUARD_STACK_PTR:
        stack_len = cfg.MODEL.NMN.STACK.LENGTH
        stack_bottom_mask = tf.one_hot(0, stack_len)
        new_stack_ptr += stack_bottom_mask * stack_ptr
    return new_stack_ptr


def _read_from_stack(att_stack, stack_ptr):
    """
    Read the value at the given stack pointer.
    """
    stack_ptr_expand = stack_ptr[:, ax, ax, :]
    # The stack pointer is a one-hot vector, so just do dot product
    att = tf.reduce_sum(att_stack * stack_ptr_expand, axis=-1, keepdims=True)
    return att


def _write_to_stack(att_stack, stack_ptr, att):
    """
    Write value 'att' into the stack at the given stack pointer. Note that the
    result needs to be assigned back to att_stack
    """
    stack_ptr_expand = stack_ptr[:, ax, ax, :]
    att_stack = att * stack_ptr_expand + att_stack * (1 - stack_ptr_expand)
    return att_stack


def _sharpen_ptr(stack_ptr):
    """
    Sharpen the stack pointers into (nearly) one-hot vectors, using argmax
    or softmax. The stack values should always sum up to one for each instance.
    """

    hard = cfg.MODEL.NMN.STACK.USE_HARD_SHARPEN
    if hard:
        # hard (non-differentiable) sharpening with argmax
        new_stack_ptr = tf.one_hot(
            tf.argmax(stack_ptr, axis=1), tf.shape(stack_ptr)[1])
    else:
        # soft (differentiable) sharpening with softmax
        temperature = cfg.MODEL.NMN.STACK.SOFT_SHARPEN_TEMP
        new_stack_ptr = tf.nn.softmax(stack_ptr / temperature)
    return new_stack_ptr


def _1x1conv(name, bottom, output_dim, reuse=None):
    return conv(name, bottom, kernel_size=1, stride=1, output_dim=output_dim,
                reuse=reuse)


def _spatial_softmax(att_raw):
    att_shape = tf.shape(att_raw)
    N = att_shape[0]
    att_softmax = tf.nn.softmax(tf.reshape(att_raw, to_T([N, -1])), axis=1)
    att_softmax = tf.reshape(att_softmax, att_shape)
    return att_softmax


def _extract_softmax_avg(kb_batch, att_raw):
    att_softmax = _spatial_softmax(att_raw)
    return tf.reduce_sum(kb_batch * att_softmax, axis=[1, 2])


def _build_module_validity_mat(module_names):
    """
    Build a module validity matrix, ensuring that only valid modules will have
    non-zero probabilities. A module is only valid to run if there are enough
    attentions to be popped from the stack, and have space to push into
    (e.g. _Find), so that stack will not underflow or overflow by design.

    module_validity_mat is a stack_len x num_module matrix, and is used to
    multiply with stack_ptr to get validity boolean vector for the modules.
    """

    stack_len = cfg.MODEL.NMN.STACK.LENGTH
    module_validity_mat = np.zeros((stack_len, len(module_names)), np.float32)
    for n_m, m in enumerate(module_names):
        # a module can be run only when stack ptr position satisfies
        # (min_ptr_pos <= ptr <= max_ptr_pos), where max_ptr_pos is inclusive
        # 1) minimum position:
        #    stack need to have MODULE_INPUT_NUM[m] things to pop from
        min_ptr_pos = MODULE_INPUT_NUM[m]
        # the stack ptr diff=(MODULE_OUTPUT_NUM[m] - MODULE_INPUT_NUM[m])
        # ensure that ptr + diff <= stack_len - 1 (stack top)
        max_ptr_pos = (
            stack_len - 1 + MODULE_INPUT_NUM[m] - MODULE_OUTPUT_NUM[m])
        module_validity_mat[min_ptr_pos:max_ptr_pos+1, n_m] = 1.

    return to_T(module_validity_mat)
