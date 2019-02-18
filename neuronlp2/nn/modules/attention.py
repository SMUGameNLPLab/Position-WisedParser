__author__ = 'max'

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class BiAAttention(nn.Module):
    '''
    Bi-Affine attention layer.
    '''

    def __init__(self, input_size_encoder, input_size_decoder, num_labels, biaffine=True, **kwargs):
        '''

        Args:
            input_size_encoder: int
                the dimension of the encoder input.
            input_size_decoder: int
                the dimension of the decoder input.
            num_labels: int
                the number of labels of the crf layer
            biaffine: bool
                if apply bi-affine parameter.
            **kwargs:
        '''
        super(BiAAttention, self).__init__()
        self.input_size_encoder = input_size_encoder
        self.input_size_decoder = input_size_decoder
        self.num_labels = num_labels
        self.biaffine = biaffine

        self.W_d = Parameter(torch.Tensor(self.num_labels, self.input_size_decoder))
        self.W_e = Parameter(torch.Tensor(self.num_labels, self.input_size_encoder))
        self.b = Parameter(torch.Tensor(self.num_labels, 1, 1))
        if self.biaffine:
            self.U = Parameter(torch.Tensor(self.num_labels, self.input_size_decoder, self.input_size_encoder))
        else:
            self.register_parameter('U', None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W_d)
        nn.init.xavier_uniform_(self.W_e)
        nn.init.constant_(self.b, 0.)
        if self.biaffine:
            nn.init.xavier_uniform_(self.U)

    def forward(self, input_d, input_e, mask_d=None, mask_e=None):
        '''

        Args:
            input_d: Tensor
                the decoder input tensor with shape = [batch, length_decoder, input_size]
            input_e: Tensor
                the child input tensor with shape = [batch, length_encoder, input_size]
            mask_d: Tensor or None
                the mask tensor for decoder with shape = [batch, length_decoder]
            mask_e: Tensor or None
                the mask tensor for encoder with shape = [batch, length_encoder]

        Returns: Tensor
            the energy tensor with shape = [batch, num_label, length, length]

        '''
        assert input_d.size(0) == input_e.size(0), 'batch sizes of encoder and decoder are requires to be equal.'
        batch, length_decoder, _ = input_d.size()
        _, length_encoder, _ = input_e.size()

        # compute decoder part: [num_label, input_size_decoder] * [batch, input_size_decoder, length_decoder]
        # the output shape is [batch, num_label, length_decoder]
        out_d = torch.matmul(self.W_d, input_d.transpose(1, 2))
        out_d=out_d.unsqueeze(3)
        # compute decoder part: [num_label, input_size_encoder] * [batch, input_size_encoder, length_encoder]
        # the output shape is [batch, num_label, length_encoder]
        out_e = torch.matmul(self.W_e, input_e.transpose(1, 2))
        out_e=out_e.unsqueeze(2)

        # output shape [batch, num_label, length_decoder, length_encoder]
        if self.biaffine:
            # compute bi-affine part
            # [batch, 1, length_decoder, input_size_decoder] * [num_labels, input_size_decoder, input_size_encoder]
            # output shape [batch, num_label, length_decoder, input_size_encoder]
            output = torch.matmul(input_d.unsqueeze(1), self.U)
            # [batch, num_label, length_decoder, input_size_encoder] * [batch, 1, input_size_encoder, length_encoder]
            # output shape [batch, num_label, length_decoder, length_encoder]
            output = torch.matmul(output, input_e.unsqueeze(1).transpose(2, 3))

            output = output + out_d + out_e + self.b
        else:
            output = out_d + out_d + self.b

        if mask_d is not None:
            output = output * mask_d.unsqueeze(1).unsqueeze(3) * mask_e.unsqueeze(1).unsqueeze(2)

        return output


class ConcatAttention(nn.Module):
    '''
    Concatenate attention layer.
    '''


    def __init__(self, input_size_encoder, input_size_decoder, hidden_size, num_labels, **kwargs):
        '''

        Args:
            input_size_encoder: int
                the dimension of the encoder input.
            input_size_decoder: int
                the dimension of the decoder input.
            hidden_size: int
                the dimension of the hidden.
            num_labels: int
                the number of labels of the crf layer
            biaffine: bool
                if apply bi-affine parameter.
            **kwargs:
        '''
        super(ConcatAttention, self).__init__()
        self.input_size_encoder = input_size_encoder
        self.input_size_decoder = input_size_decoder
        self.hidden_size = hidden_size
        self.num_labels = num_labels

        self.W_d = Parameter(torch.Tensor(self.input_size_decoder, self.hidden_size))
        self.W_e = Parameter(torch.Tensor(self.input_size_encoder, self.hidden_size))
        self.b = Parameter(torch.Tensor(self.hidden_size))
        self.v = Parameter(torch.Tensor(self.hidden_size, self.num_labels))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform(self.W_d)
        nn.init.xavier_uniform(self.W_e)
        nn.init.xavier_uniform(self.v)
        nn.init.constant(self.b, 0.)

    def forward(self, input_d, input_e, mask_d=None, mask_e=None):
        '''

        Args:
            input_d: Tensor
                the decoder input tensor with shape = [batch, length_decoder, input_size]
            input_e: Tensor
                the child input tensor with shape = [batch, length_encoder, input_size]
            mask_d: Tensor or None
                the mask tensor for decoder with shape = [batch, length_decoder]
            mask_e: Tensor or None
                the mask tensor for encoder with shape = [batch, length_encoder]

        Returns: Tensor
            the energy tensor with shape = [batch, num_label, length, length]

        '''
        assert input_d.size(0) == input_e.size(0), 'batch sizes of encoder and decoder are requires to be equal.'
        batch, length_decoder, _ = input_d.size()
        _, length_encoder, _ = input_e.size()

        # compute decoder part: [batch, length_decoder, input_size_decoder] * [input_size_decoder, hidden_size]
        # the output shape is [batch, length_decoder, hidden_size]
        # then --> [batch, 1, length_decoder, hidden_size]
        out_d = torch.matmul(input_d, self.W_d).unsqueeze(1)
        # compute decoder part: [batch, length_encoder, input_size_encoder] * [input_size_encoder, hidden_size]
        # the output shape is [batch, length_encoder, hidden_size]
        # then --> [batch, length_encoder, 1, hidden_size]
        out_e = torch.matmul(input_e, self.W_e).unsqueeze(2)

        # add them together [batch, length_encoder, length_decoder, hidden_size]
        out = F.tanh(out_d + out_e + self.b)

        # product with v
        # [batch, length_encoder, length_decoder, hidden_size] * [hidden, num_label]
        # [batch, length_encoder, length_decoder, num_labels]
        # then --> [batch, num_labels, length_decoder, length_encoder]
        return torch.matmul(out, self.v).transpose(1, 3)

class MultiHeadAttention(nn.Module):
    def __init__(self,input_size_encoder, input_size_decoder, num_labels, n_heads):
        super().__init__()

        self.input_size_encoder = input_size_encoder
        self.input_size_decoder = input_size_decoder

        self.num_labels = num_labels
        self.n_heads = n_heads

        self.d_k = int(self.input_size_decoder/n_heads)
        self.d_q = int(self.input_size_encoder/n_heads)
        self.d_v = int(self.input_size_encoder/n_heads)

        self.W_Q = Parameter(torch.Tensor(n_heads,self.input_size_decoder,self.d_q))
        self.W_K = Parameter(torch.Tensor(n_heads,self.input_size_decoder,self.d_k))
        self.W_V = Parameter(torch.Tensor(n_heads,self.input_size_decoder,self.d_v))

        self.W_O = Parameter(torch.Tensor(self.d_v * n_heads,num_labels))
        # self.activation = F.tanh
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W_Q)
        nn.init.xavier_uniform_(self.W_K)
        nn.init.xavier_uniform_(self.W_V)
        nn.init.xavier_uniform_(self.W_O)

    def forward(self, input_d, input_e, mask_d=None, mask_e=None):

        q_i = torch.matmul(input_d.unsqueeze(1),self.W_Q)
        k_i = torch.matmul(input_e.unsqueeze(1),self.W_K)
        v_i = torch.matmul(input_e.unsqueeze(1),self.W_V)

        heads = self.scaled_dot_product(q_i,k_i,v_i)
        heads = heads.transpose(1,3)
        heads = heads.reshape(*heads.shape[:-2],-1)
        heads = heads.transpose(1,2)

        heads += input_d.unsqueeze(2)


        result = torch.matmul(heads,self.W_O).squeeze(-1)
        if mask_d is not None:
            result = result * mask_d.unsqueeze(2) * mask_e.unsqueeze(1)
        return result

    def scaled_dot_product(self,q,k,v):
        t = torch.sum((q.unsqueeze(3) * k.unsqueeze(2)),4) /self.d_k
        # t = torch.matmul(q,k.transpose(1,2)) / self.d_k
        t = F.softmax(t,dim=3)
        return t.unsqueeze(-1)*v.unsqueeze(2)
