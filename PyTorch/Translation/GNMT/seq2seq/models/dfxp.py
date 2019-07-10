import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.modules.rnn as rnn
from torch.nn.utils.rnn import PackedSequence
from torch.utils.cpp_extension import load


dfxp_backend = load(name='dfxp_backend',
    sources=['seq2seq/models/dfxp.cpp', 'seq2seq/models/dfxp.cu'],
    extra_cflags=['-std=c++11', '-O3', '-D_GLIBCXX_USE_CXX11_ABI=1', '-L/usr/local/cuda-9.0/lib64', '-v'],
    extra_cuda_cflags=['--use_fast_math', '-L/usr/local/cuda-9.0/lib64', '-v'],
    extra_ldflags=['-lcurand'],
    verbose=True, build_directory='.')


class Quantize(torch.autograd.Function):

    @staticmethod
    def forward(ctx, X, qmin, qmax, step, update_step):
        return dfxp_backend.dfxp_quantize_forward(X, qmin, qmax, step, update_step)

    @staticmethod
    def backward(ctx, grad):
        return grad, None, None, None, None


class GradQuantize(torch.autograd.Function):

    @staticmethod
    def forward(ctx, X, qmin, qmax, step, update_step):
        ctx.save_for_backward(qmin, qmax, step)
        ctx.update_step = update_step
        return X

    @staticmethod
    def backward(ctx, grad):
        qmin, qmax, step = ctx.saved_tensors
        update_step = ctx.update_step
        grad = dfxp_backend.dfxp_grad_quantize_backward(grad, qmin, qmax, step, update_step)
        return grad, None, None, None, None


class ForwardQuantizer(nn.Module):

    def __init__(self, bits, step=2.0 ** -5):
        super().__init__()

        if bits == 32:
            self.forward = self.identity
        else:
            self.forward = self.forward_q

        self.register_buffer('qmin', torch.tensor(-(2.0 ** (bits - 1))))
        self.register_buffer('qmax', torch.tensor(2.0 ** (bits - 1) - 1))
        self.register_buffer('step', torch.tensor(step))

        self.update_step = False

    def forward_q(self, X):
        return Quantize.apply(X, self.qmin, self.qmax, self.step, self.update_step)

    def identity(self, X):
        return X

    def cuda(self):
        super().cuda()

        # move back to cpu for faster dereferencing
        self.qmin.cpu()
        self.qmax.cpu()
        self.step.cpu()


class BackwardQuantizer(nn.Module):

    def __init__(self, bits, step=2.0 ** -5):
        super().__init__()

        if bits == 32:
            self.forward = self.identity
        else:
            self.forward = self.forward_q

        self.register_buffer('qmin', torch.tensor(-(2.0 ** (bits - 1))))
        self.register_buffer('qmax', torch.tensor(2.0 ** (bits - 1) - 1))
        self.register_buffer('step', torch.tensor(step))

        self.update_step = False

    def forward_q(self, X):
        return GradQuantize.apply(X, self.qmin, self.qmax, self.step, self.update_step)

    def identity(self, X):
        return X

    def cuda(self):
        super().cuda()

        # move back to cpu for faster dereferencing
        self.qmin.cpu()
        self.qmax.cpu()
        self.step.cpu()


class Conv2d_q(nn.Module):

    def __init__(self, bits, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()

        def pair(x):
            try:
                for _ in x:
                    pass
                return x
            except TypeError:
                return [x, x]

        self.bits = bits
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = pair(kernel_size)
        self.stride = pair(stride)
        self.padding = pair(padding)

        self.weight = nn.Parameter(torch.Tensor(
            out_channels, in_channels,
            self.kernel_size[0], self.kernel_size[1]))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

        self.input_q = ForwardQuantizer(bits)
        self.weight_q = ForwardQuantizer(bits)
        if self.bias is not None:
            self.bias_q = ForwardQuantizer(bits)
        else:
            self.bias_q = lambda x: x
        self.grad_q = BackwardQuantizer(bits)

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, X):
        X = self.input_q(X)
        weight = self.weight_q(self.weight)
        bias = self.bias_q(self.bias)
        out = F.conv2d(X, weight, bias, self.stride, self.padding)
        out = self.grad_q(out)
        return out

    def extra_repr(self):
        s = ('{bits}bits, {in_channels}, {out_channels}, '
             'kernel_size={kernel_size}, stride={stride}')
        return s.format(**self.__dict__)


class Linear_q(nn.Module):

    def __init__(self, bits, in_features, out_features, bias=True):
        super().__init__()

        self.bits = bits
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

        self.input_q = ForwardQuantizer(bits)
        self.weight_q = ForwardQuantizer(bits)
        if self.bias is not None:
            self.bias_q = ForwardQuantizer(bits)
        else:
            self.bias_q = lambda x: x
        self.grad_q = BackwardQuantizer(bits)

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, X):
        X = self.input_q(X)
        weight = self.weight_q(self.weight)
        bias = self.bias_q(self.bias)
        out = F.linear(X, weight, bias)
        out = self.grad_q(out)
        return out

    def extra_repr(self):
        s = '{bits}bits, {in_features}, {out_features}'
        return s.format(**self.__dict__)


class Normalize2d_q(nn.Module):

    def __init__(self, bits, num_features, eps=1e-5, momentum=0.1):
        super().__init__()

        self.bit = bits
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.reset_parameters()

        self.input_q = ForwardQuantizer(bits)
        self.grad_q = BackwardQuantizer(bits)

    def reset_parameters(self):
        self.running_mean.zero_()
        self.running_var.fill_(1)

    def forward(self, X):
        X = self.input_q(X)
        out = F.batch_norm(X, self.running_mean, self.running_var,
            weight=None, bias=None, training=self.training,
            momentum=self.momentum, eps=self.eps)
        out = self.grad_q(out)
        return out


class Rescale2d_q(nn.Module):

    def __init__(self, bits, num_features):
        super().__init__()

        self.bits = bits
        self.num_features = num_features

        self.weight = nn.Parameter(torch.Tensor(num_features, 1, 1))
        self.bias = nn.Parameter(torch.Tensor(num_features, 1, 1))
        self.reset_parameters()

        self.input_q = ForwardQuantizer(bits)
        self.weight_q = ForwardQuantizer(bits)
        self.bias_q = ForwardQuantizer(bits)
        self.grad_q = BackwardQuantizer(bits)

    def reset_parameters(self):
        nn.init.ones_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, X):
        X = self.input_q(X)
        weight = self.weight_q(self.weight)
        bias = self.bias_q(self.bias)
        out = X * weight + bias
        out = self.grad_q(out)
        return out


class BatchNorm2d_q(nn.Module):

    def __init__(self, bits, num_features, eps=1e-5, momentum=0.1, affine=True):
        super().__init__()

        self.bits = bits
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine

        self.normalize = Normalize2d_q(bits, num_features, eps, momentum)
        if affine:
            self.rescale = Rescale2d_q(bits, num_features)
            self.weight = self.rescale.weight
            self.bias = self.rescale.bias
        else:
            self.rescale = lambda x: x

    def forward(self, X):
        out = self.normalize(X)
        out = self.rescale(out)
        return out

    def extra_repr(self):
        s = '{bits}bits, {num_features}, eps={eps}, momentum={momentum}, affine={affine}'
        return s.format(**self.__dict__)


class LSTMCell_q(nn.Module):

    def __init__(self, bits, input_size, hidden_size, bias=True):
        super().__init__()

        self.bits = bits
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias

        # parameters
        self.weight = nn.Parameter(torch.Tensor(4 * hidden_size, input_size + hidden_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(4 * hidden_size))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

        # forward quantizers
        self.input_q = ForwardQuantizer(bits)
        self.weight_q = ForwardQuantizer(bits)
        if self.bias is not None:
            self.bias_q = ForwardQuantizer(bits)
        else:
            self.bias_q = lambda x: x
        self.cell_q = ForwardQuantizer(bits)
        self.fgate_q = ForwardQuantizer(bits)
        self.igate_q = ForwardQuantizer(bits)
        self.ggate_q = ForwardQuantizer(bits)
        self.ogate_q = ForwardQuantizer(bits)
        self.cell_tanh_q = ForwardQuantizer(bits)

        # backward quantizers
        self.matmul_grad_q = BackwardQuantizer(bits)
        self.fgate_grad_q = BackwardQuantizer(bits)
        self.igate_grad_q = BackwardQuantizer(bits)
        self.ggate_grad_q = BackwardQuantizer(bits)
        self.ogate_grad_q = BackwardQuantizer(bits)
        self.cf_grad_q = BackwardQuantizer(bits)
        self.ig_grad_q = BackwardQuantizer(bits)
        self.cell_tanh_grad_q = BackwardQuantizer(bits)
        self.hidden_grad_q = BackwardQuantizer(bits)

    def reset_parameters(self):
        stdv = self.hidden_size ** -0.5
        for weight in self.parameters():
            nn.init.uniform_(weight, -stdv, stdv)

    def forward(self, X, hx=None):
        if hx is None:
            hx = X.new_zeros(X.size(0), self.hidden_size, requires_grad=False)
            hx = (hx, hx)

        h, c = hx
        X = torch.cat((X, h), 1)
        X = self.input_q(X)
        weight = self.weight_q(self.weight)
        bias = self.bias_q(self.bias)
        y = F.linear(X, weight, bias)
        y = self.matmul_grad_q(y)

        i, f, g, o = y.chunk(4, 1)

        i = F.sigmoid(i)
        f = F.sigmoid(f)
        g = F.tanh(g)
        o = F.sigmoid(o)

        i = self.igate_q(self.igate_grad_q(i))
        f = self.fgate_q(self.fgate_grad_q(f))
        g = self.ggate_q(self.ggate_grad_q(g))
        o = self.ogate_q(self.ogate_grad_q(o))

        c = self.cell_q(c)
        c = self.cf_grad_q(c * f) + self.ig_grad_q(i * g)

        cell_tanh = F.tanh(c)
        cell_tanh = self.cell_tanh_q(self.cell_tanh_grad_q(cell_tanh))
        h = self.hidden_grad_q(cell_tanh * o)
        return (h, c)
        

    def extra_repr(self):
        s = '{bits}bits, {input_size}, {hidden_size}'
        return s.format(**self.__dict__)


class LSTM_q(nn.Module):

    def __init__(self, bits, input_size, hidden_size,
                 num_layers=1, bias=True, batch_first=False,
                 dropout=0, bidirectional=False):
        super().__init__()

        # this is not compatible with nn.LSTM, but sufficient for GNMT
        assert num_layers == 1, f'LSTM_q requires num_layers == 1 but found {num_layers}'
        assert not batch_first, f'LSTM_q requires batch_first == False but found {batch_first}'
        assert dropout == 0, f'LSTM_q requires dropout == 0 but found {dropout}'

        self.bits = bits
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.batch_first = batch_first
        self.bidirectional = bidirectional

        self.cell = LSTMCell_q(bits, input_size, hidden_size)
        if bidirectional:
            self.cell_reverse = LSTMCell_q(bits, input_size, hidden_size)

    def forward(self, X, hx=None):
        is_packed = isinstance(X, PackedSequence)
        assert not self.bidirectional or is_packed, (
            'Input of bidirectional LSTM must be packed.')

        if is_packed:
            X, batch_sizes = X
            output, hx = self.forward_packed_sequence(X, batch_sizes, hx)
            output_reversed, hx_reversed = self.forward_packed_sequence_reversed(
                X, batch_sizes, hx)
            output = torch.cat((output, output_reversed), 1)
            hx = tuple(torch.cat((h, h_reversed), 1)
                for h, h_reversed in zip(hx, hx_reversed))
            return PackedSequence(output, batch_sizes), hx
        else:
            return self.forward_tensor(X, hx)

    def forward_tensor(self, X, hx):
        output = []
        seq_len = X.size(0)
        for X_step in X:
            hx = self.cell(X_step, hx)
            output.append(hx[0])

        # shape: (seq_len * batch_size, hidden_size)
        output = torch.cat(tuple(output), 0)
        output = output.view(seq_len, -1, self.hidden_size)
        return output, hx

    def forward_packed_sequence(self, X, batch_sizes, hx):
        output = []
        hiddens = []
        input_offset = 0
        last_batch_size = batch_sizes[0]
        for batch_size in batch_sizes:
            X_step = X[input_offset:input_offset + batch_size]
            input_offset += batch_size

            # split hx
            dec = last_batch_size - batch_size
            last_batch_size = batch_size
            if dec > 0:
                hiddens.append(tuple(h[-dec:] for h in hx))
                hx = tuple(h[:-dec] for h in hx)

            # cell step
            hx = self.cell(X_step, hx)
            output.append(hx[0])

        hiddens.append(hx)
        hiddens.reverse() # last instances are shorter

        hx = tuple(torch.cat(h, 0) for h in zip(*hiddens))
        output = torch.cat(tuple(output), 0)
        return output, hx
        

    def forward_packed_sequence_reversed(self, X, batch_sizes, hx):
        if hx is None:
            hx = X.new_zeros(X.size(0), self.hidden_size, requires_grad=False)
            hx = (hx, hx)

        output = []
        hiddens = []
        input_offset = X.size(0)
        last_batch_size = batch_sizes[-1]
        initial_hx = hx
        hx = tuple(h[:last_batch_size] for h in hx)
        for batch_size in reversed(batch_sizes):
            X_step = X[input_offset - batch_size:input_offset]
            input_offset -= batch_size

            # append hx
            inc = batch_size - last_batch_size
            if inc > 0:
                hx = tuple(torch.cat((h, ih[last_batch_size:batch_size]), 0)
                    for h, ih in zip(hx, initial_hx))
            last_batch_size = batch_size

            # cell step
            hx = self.cell_reverse(X_step, hx)
            output.append(hx[0])

        output.reverse()
        output = torch.cat(tuple(output), 0)
        return output, hx

    def extra_repr(self):
        s = ('{bits}bits, {input_size}, {hidden_size}, '
             'bidirectional={bidirectional}')
        return s.format(**self.__dict__)
