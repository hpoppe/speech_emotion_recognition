import torch
import torch.nn as nn
import torchaudio.transforms as T
import torch.nn.functional as F
import math
from torch.nn.utils import weight_norm
import copy
import numpy as np
from transformers import BertTokenizer, AutoModelForSequenceClassification, AutoProcessor, HubertModel

# the model structure with BERT is not done

# ESPnet https://github.com/espnet/espnet/tree/master
class LayerNorm(torch.nn.LayerNorm):
    """Layer normalization module.
    Args:
        nout (int): Output dim size.
        dim (int): Dimension to be normalized.
    """

    def __init__(self, nout, dim=-1):
        """Construct an LayerNorm object."""
        super(LayerNorm, self).__init__(nout, eps=1e-12)
        self.dim = dim

    def forward(self, x):
        """Apply layer normalization.
        Args:
            x (torch.Tensor): Input tensor.
        Returns:
            torch.Tensor: Normalized tensor.
        """
        if self.dim == -1:
            return super(LayerNorm, self).forward(x)
        return (
            super(LayerNorm, self)
            .forward(x.transpose(self.dim, -1))
            .transpose(self.dim, -1)
        )
    

class EncoderLayer(nn.Module):
    """Encoder layer module.
    Args:
        size (int): Input dimension.
        self_attn (torch.nn.Module): Self-attention module instance.
            `MultiHeadedAttention` or `RelPositionMultiHeadedAttention` instance
            can be used as the argument.
        feed_forward (torch.nn.Module): Feed-forward module instance.
            `PositionwiseFeedForward`, `MultiLayeredConv1d`, or `Conv1dLinear` instance
            can be used as the argument.
        dropout_rate (float): Dropout rate.
        normalize_before (bool): Whether to use layer_norm before the first block.
        concat_after (bool): Whether to concat attention layer's input and output.
            if True, additional linear will be applied.
            i.e. x -> x + linear(concat(x, att(x)))
            if False, no additional linear will be applied. i.e. x -> x + att(x)
        stochastic_depth_rate (float): Proability to skip this layer.
            During training, the layer may skip residual computation and return input
            as-is with given probability.
    """

    def __init__(
        self,
        size,
        self_attn,
        feed_forward,
        dropout_rate,
        normalize_before=True,
        concat_after=False,
        stochastic_depth_rate=0.0,
    ):
        """Construct an EncoderLayer object."""
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.norm1 = LayerNorm(size)
        self.norm2 = LayerNorm(size)
        self.dropout = nn.Dropout(dropout_rate)
        self.size = size
        self.normalize_before = normalize_before
        self.concat_after = concat_after
        if self.concat_after:
            self.concat_linear = nn.Linear(size + size, size)
        self.stochastic_depth_rate = stochastic_depth_rate

    def forward(self, x, mask, cache=None):
        """Compute encoded features.
        Args:
            x_input (torch.Tensor): Input tensor (#batch, time, size).
            mask (torch.Tensor): Mask tensor for the input (#batch, 1, time).
            cache (torch.Tensor): Cache tensor of the input (#batch, time - 1, size).
        Returns:
            torch.Tensor: Output tensor (#batch, time, size).
            torch.Tensor: Mask tensor (#batch, 1, time).
        """
        skip_layer = False
        # with stochastic depth, residual connection `x + f(x)` becomes
        # `x <- x + 1 / (1 - p) * f(x)` at training time.
        stoch_layer_coeff = 1.0
        if self.training and self.stochastic_depth_rate > 0:
            skip_layer = torch.rand(1).item() < self.stochastic_depth_rate
            stoch_layer_coeff = 1.0 / (1 - self.stochastic_depth_rate)

        if skip_layer:
            if cache is not None:
                x = torch.cat([cache, x], dim=1)
            return x, mask

        residual = x
        if self.normalize_before:
            x = self.norm1(x)

        if cache is None:
            x_q = x
        else:
            assert cache.shape == (x.shape[0], x.shape[1] - 1, self.size)
            x_q = x[:, -1:, :]
            residual = residual[:, -1:, :]
            mask = None if mask is None else mask[:, -1:, :]

        if self.concat_after:
            x_concat = torch.cat((x, self.self_attn(x_q, x, x, mask)), dim=-1)
            x = residual + stoch_layer_coeff * self.concat_linear(x_concat)
        else:
            x = residual + stoch_layer_coeff * self.dropout(
                self.self_attn(x_q, x, x, mask)
            )
        '''if self.concat_after:
            x_concat = torch.cat((x, self.self_attn(x)), dim=-1)
            x = residual + stoch_layer_coeff * self.concat_linear(x_concat)
        else:
            x = residual + stoch_layer_coeff * self.dropout(
                self.self_attn(x)
            )'''
        if not self.normalize_before:
            x = self.norm1(x)

        residual = x
        if self.normalize_before:
            x = self.norm2(x)
        x = residual + stoch_layer_coeff * self.dropout(self.feed_forward(x))
        if not self.normalize_before:
            x = self.norm2(x)

        if cache is not None:
            x = torch.cat([cache, x], dim=1)

        return x, mask


# Happy Color https://github.com/HappyColor/
class Speech_MultiHeadedAttention(nn.Module):
    ''' Speech-based Multi-Head Self-Attention (Speech-MSA)
    
    Input dimension order is (batch_size, seq_len, input_dim).
    '''
    def __init__(self, d_model, n_heads, local_size, dropout=0.2, bias=True, overlap=False):
        super(Speech_MultiHeadedAttention, self).__init__()
        self.qdim = d_model
        self.kdim = d_model
        self.vdim = d_model
        self.local_size = int(local_size)
        self.overlap = overlap    #  overlap = True may have nondeterministic behavior.

        self.project_qkv = nn.Linear(d_model, 3 * d_model, bias=bias)
        self.project_out = nn.Linear(d_model, d_model, bias=bias)

        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.head_dim = d_model // n_heads
        self.n_heads = n_heads
        self.d_model = d_model
        self.dropout = dropout
        self.scaling = float(self.head_dim) ** -0.5
    
    def get_overlap_segments(self, x: torch.Tensor, window_size: int):
        '''Get overlap segments for local attention.
        Args: 
            x: Input sequence in shape (B, T, C).
            window_size: The needed length of the segment. Must be an odd number.
        '''
        # assert window_size % 2, f'window_size must be an odd number, but get {window_size}.'
        if not window_size % 2:
            window_size += 1     # window_size must be an odd number
        
        b, t, c = x.shape
        pad_len = (window_size - 1) // 2
        x = F.pad(x, (0, 0, pad_len, pad_len), value=0)

        stride = x.stride()
        out_shape = (b, t, window_size, c)
        out_stride = (stride[0], stride[1], stride[1], stride[2])

        return torch.as_strided(x, size=out_shape, stride=out_stride)

    def forward(self, x):
        '''
        Args:
            key_padding_mask: if provided, specified padding elements in the key will
                be ignored by the attention. This is an binary mask. When the value is True,
                the corresponding value on the attention layer will be filled with -inf.
            attn_mask: mask that prevents attention to certain positions. This is an additive mask
                (i.e. the values will be added to the attention layer).
        Shape:
            Inputs:
            - x: :math:`(B, T, E)` where T is the target sequence length, B is the batch size, E is
              the embedding dimension.
            - key_padding_mask: :math:`(B, S)`, ByteTensor, where B is the batch size, S is the source sequence length.
              3-D key_padding_mask with math:`(B, T, S)` is supported now, where T is the target sequence length.
            - attn_mask: :math:`(T, S)` where T is the target sequence length, S is the source sequence length.
        '''
        bsz, tgt_len = x.shape[:2]

        if self.local_size == -1:
            local_size = tgt_len
            global_attn = True
        else:
            local_size = self.local_size
            global_attn = False

        if not self.overlap:
            need_pad = tgt_len % local_size
            if need_pad:
                pad = local_size - need_pad
                x = F.pad(x, (0, 0, 0, pad), mode='constant', value=0)
                tgt_len += pad
        else:
            need_pad = 0

        Q, K, V = self.project_qkv(x).chunk(3, dim=-1)
        Q = Q * self.scaling
        Q = Q.transpose(0, 1).contiguous().view(tgt_len, bsz * self.n_heads, self.head_dim).transpose(0, 1)
        K = K.transpose(0, 1).contiguous().view(-1, bsz * self.n_heads, self.head_dim).transpose(0, 1)
        V = V.transpose(0, 1).contiguous().view(-1, bsz * self.n_heads, self.head_dim).transpose(0, 1)

        if (self.overlap) and (not global_attn):
            Q = Q.unsqueeze(dim=2)
            K = self.get_overlap_segments(K, window_size=local_size).transpose(-1, -2)
            V = self.get_overlap_segments(V, window_size=local_size)

            attn_output_weights = torch.matmul(Q, K)
            attn_output_weights = F.softmax(attn_output_weights, dim=-1)
            attn_output_weights = F.dropout(attn_output_weights, p=self.dropout, training=self.training)

            attn_output = torch.matmul(attn_output_weights, V).squeeze(dim=2)
        else:
            Q = Q.contiguous().view(-1, local_size, self.head_dim)
            K = K.contiguous().view(-1, local_size, self.head_dim)
            V = V.contiguous().view(-1, local_size, self.head_dim)

            src_len = K.size(1)
            attn_output_weights = torch.bmm(Q, K.transpose(1, 2))

            assert list(attn_output_weights.size()) == [bsz * self.n_heads * tgt_len / local_size, local_size, src_len]

            attn_output_weights = F.softmax(attn_output_weights, dim=-1)
            attn_output_weights = F.dropout(attn_output_weights, p=self.dropout, training=self.training)

            attn_output = torch.bmm(attn_output_weights, V)

            assert list(attn_output.size()) == [bsz * self.n_heads * tgt_len / local_size, local_size, self.head_dim]
            attn_output = attn_output.view(bsz * self.n_heads, tgt_len, self.head_dim)

        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, self.d_model).transpose(0, 1)
        attn_output = self.project_out(attn_output)

        if need_pad:
            attn_output = attn_output[:, :-pad, :]

        return attn_output


class SMultiHeadedAttention(nn.Module):
    def __init__(self, n_head, n_feat, dropout_rate):
        super(SMultiHeadedAttention, self).__init__()
        assert n_feat % n_head == 0
        self.d_k = n_feat // n_head
        self.h = n_head
        self.linear_qkv = nn.Linear(n_feat*3, n_feat*3)  # Multiply by 3 for Q, K, V
        self.linear_out = nn.Linear(n_feat, n_feat)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward_qkv(self, query, key, value):
        n_batch = query.size(0)
        # concatenate along the feature dimension
        qkv = torch.cat([query, key, value], dim=-1)  # (batch, time, size*3)
        # linear transformation
        qkv = self.linear_qkv(qkv).view(n_batch, -1, self.h, 3*self.d_k)  # (batch, time, head, d_k*3)
        # split back into query, key, value
        q, k, v = torch.split(qkv, self.d_k, dim=-1)  # (batch, time, head, d_k) each
        q = q.transpose(1, 2)  # (batch, head, time1, d_k)
        k = k.transpose(1, 2)  # (batch, head, time2, d_k)
        v = v.transpose(1, 2)  # (batch, head, time2, d_k)

        return q, k, v
    

    def forward_attention(self, value, scores, mask):
        """Compute attention context vector.
        Args:
            value (torch.Tensor): Transformed value (#batch, n_head, time2, d_k).
            scores (torch.Tensor): Attention score (#batch, n_head, time1, time2).
            mask (torch.Tensor): Mask (#batch, 1, time2) or (#batch, time1, time2).
        Returns:
            torch.Tensor: Transformed value (#batch, time1, d_model)
                weighted by the attention score (#batch, time1, time2).
        """
        n_batch = value.size(0)
        if mask is not None:
            mask = mask.unsqueeze(1).eq(0)  # (batch, 1, *, time2)
            min_value = torch.finfo(scores.dtype).min
            scores = scores.masked_fill(mask, min_value)
            self.attn = torch.softmax(scores, dim=-1).masked_fill(
                mask, 0.0
            )  # (batch, head, time1, time2)
        else:
            self.attn = torch.softmax(scores, dim=-1)  # (batch, head, time1, time2)

        p_attn = self.dropout(self.attn)
        x = torch.matmul(p_attn, value)  # (batch, head, time1, d_k)
        x = (
            x.transpose(1, 2).contiguous().view(n_batch, -1, self.h * self.d_k)
        )  # (batch, time1, d_model)

        return self.linear_out(x)  # (batch, time1, d_model)

    def forward(self, query, key, value, mask):
        """Compute scaled dot product attention.
        Args:
            query (torch.Tensor): Query tensor (#batch, time1, size).
            key (torch.Tensor): Key tensor (#batch, time2, size).
            value (torch.Tensor): Value tensor (#batch, time2, size).
            mask (torch.Tensor): Mask tensor (#batch, 1, time2) or
                (#batch, time1, time2).
        Returns:
            torch.Tensor: Output tensor (#batch, time1, d_model).
        """
        q, k, v = self.forward_qkv(query, key, value)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        return self.forward_attention(v, scores, mask)
    

class MultiHeadedAttention(nn.Module):
    """Multi-Head Attention layer.
    Args:
        n_head (int): The number of heads.
        n_feat (int): The number of features.
        dropout_rate (float): Dropout rate.
    """

    def __init__(self, n_head, n_feat, dropout_rate):
        """Construct an MultiHeadedAttention object."""
        super(MultiHeadedAttention, self).__init__()
        assert n_feat % n_head == 0
        # We assume d_v always equals d_k
        self.d_k = n_feat // n_head
        self.h = n_head
        self.linear_q = nn.Linear(n_feat, n_feat)
        self.linear_k = nn.Linear(n_feat, n_feat)
        self.linear_v = nn.Linear(n_feat, n_feat)
        self.linear_out = nn.Linear(n_feat, n_feat)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward_qkv(self, query, key, value):
        """Transform query, key and value.
        Args:
            query (torch.Tensor): Query tensor (#batch, time1, size).
            key (torch.Tensor): Key tensor (#batch, time2, size).
            value (torch.Tensor): Value tensor (#batch, time2, size).
        Returns:
            torch.Tensor: Transformed query tensor (#batch, n_head, time1, d_k).
            torch.Tensor: Transformed key tensor (#batch, n_head, time2, d_k).
            torch.Tensor: Transformed value tensor (#batch, n_head, time2, d_k).
        """
        n_batch = query.size(0)
        q = self.linear_q(query).view(n_batch, -1, self.h, self.d_k)
        k = self.linear_k(key).view(n_batch, -1, self.h, self.d_k)
        v = self.linear_v(value).view(n_batch, -1, self.h, self.d_k)
        q = q.transpose(1, 2)  # (batch, head, time1, d_k)
        k = k.transpose(1, 2)  # (batch, head, time2, d_k)
        v = v.transpose(1, 2)  # (batch, head, time2, d_k)

        return q, k, v

    def forward_attention(self, value, scores, mask):
        """Compute attention context vector.
        Args:
            value (torch.Tensor): Transformed value (#batch, n_head, time2, d_k).
            scores (torch.Tensor): Attention score (#batch, n_head, time1, time2).
            mask (torch.Tensor): Mask (#batch, 1, time2) or (#batch, time1, time2).
        Returns:
            torch.Tensor: Transformed value (#batch, time1, d_model)
                weighted by the attention score (#batch, time1, time2).
        """
        n_batch = value.size(0)
        if mask is not None:
            mask = mask.unsqueeze(1).eq(0)  # (batch, 1, *, time2)
            min_value = torch.finfo(scores.dtype).min
            scores = scores.masked_fill(mask, min_value)
            self.attn = torch.softmax(scores, dim=-1).masked_fill(
                mask, 0.0
            )  # (batch, head, time1, time2)
        else:
            self.attn = torch.softmax(scores, dim=-1)  # (batch, head, time1, time2)

        p_attn = self.dropout(self.attn)
        x = torch.matmul(p_attn, value)  # (batch, head, time1, d_k)
        x = (
            x.transpose(1, 2).contiguous().view(n_batch, -1, self.h * self.d_k)
        )  # (batch, time1, d_model)

        return self.linear_out(x)  # (batch, time1, d_model)

    def forward(self, query, key, value, mask):
        """Compute scaled dot product attention.
        Args:
            query (torch.Tensor): Query tensor (#batch, time1, size).
            key (torch.Tensor): Key tensor (#batch, time2, size).
            value (torch.Tensor): Value tensor (#batch, time2, size).
            mask (torch.Tensor): Mask tensor (#batch, 1, time2) or
                (#batch, time1, time2).
        Returns:
            torch.Tensor: Output tensor (#batch, time1, d_model).
        """
        q, k, v = self.forward_qkv(query, key, value)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        return self.forward_attention(v, scores, mask)
    

def create_PositionalEncoding(input_dim, max_seq_len=2000): 
    position_encoding = np.array([ 
        [pos / np.power(10000, 2.0 * (j // 2) / input_dim) for j in range(input_dim)] 
        for pos in range(max_seq_len)]) 
    
    position_encoding[:, 0::2] = np.sin(position_encoding[:, 0::2])
    position_encoding[:, 1::2] = np.cos(position_encoding[:, 1::2])

    position_encoding = torch.from_numpy(position_encoding.astype(np.float32))
    position_encoding = nn.Parameter(position_encoding, requires_grad=False) 
    
    return position_encoding


# ESPnet https://github.com/espnet/espnet/tree/master
class PositionalEncoding(torch.nn.Module):
    """Positional encoding.
    Args:
        d_model (int): Embedding dimension.
        dropout_rate (float): Dropout rate.
        max_len (int): Maximum input length.
        reverse (bool): Whether to reverse the input position. Only for
        the class LegacyRelPositionalEncoding. We remove it in the current
        class RelPositionalEncoding.
    """

    def __init__(self, d_model, dropout_rate, max_len=5000, reverse=False):
        """Construct an PositionalEncoding object."""
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.reverse = reverse
        self.xscale = math.sqrt(self.d_model)
        self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.pe = None
        self.extend_pe(torch.tensor(0.0).expand(1, max_len))
        # self._register_load_state_dict_pre_hook(_pre_hook)

    def extend_pe(self, x):
        """Reset the positional encodings."""
        if self.pe is not None:
            if self.pe.size(1) >= x.size(1):
                if self.pe.dtype != x.dtype or self.pe.device != x.device:
                    self.pe = self.pe.to(dtype=x.dtype, device=x.device)
                return
        pe = torch.zeros(x.size(1), self.d_model)
        if self.reverse:
            position = torch.arange(
                x.size(1) - 1, -1, -1.0, dtype=torch.float32
            ).unsqueeze(1)
        else:
            position = torch.arange(0, x.size(1), dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32)
            * -(math.log(10000.0) / self.d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.pe = pe.to(device=x.device, dtype=x.dtype)

    def forward(self, x: torch.Tensor):
        """Add positional encoding.
        Args:
            x (torch.Tensor): Input tensor (batch, time, `*`).
        Returns:
            torch.Tensor: Encoded tensor (batch, time, `*`).
        """
        self.extend_pe(x)
        x = x * self.xscale + self.pe[:, : x.size(1)]
        return self.dropout(x)
    

class Conv2dSubsampling(torch.nn.Module):
    """Convolutional 2D subsampling (to 1/4 length).
    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.
        pos_enc (torch.nn.Module): Custom position encoding layer.
    """

    def __init__(self, idim, odim, dropout_rate, pos_enc=None):
        """Construct an Conv2dSubsampling object."""
        super(Conv2dSubsampling, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, odim, 3, 2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(odim, odim, 3, 2),
            torch.nn.ReLU(),
        )
        self.out = torch.nn.Sequential(
            torch.nn.Linear(odim * (((idim - 1) // 2 - 1) // 2), odim),
            pos_enc if pos_enc is not None else PositionalEncoding(odim, dropout_rate),
        )

    def forward(self, x, x_mask):
        """Subsample x.
        Args:
            x (torch.Tensor): Input tensor (#batch, time, idim).
            x_mask (torch.Tensor): Input mask (#batch, 1, time).
        Returns:
            torch.Tensor: Subsampled tensor (#batch, time', odim),
                where time' = time // 4.
            torch.Tensor: Subsampled mask (#batch, 1, time'),
                where time' = time // 4.
        """
        x = x.unsqueeze(1)  # (b, c, t, f)
        # print(x.size())
        x = self.conv(x)
        b, c, t, f = x.size()
        x = self.out(x.transpose(1, 2).contiguous().view(b, t, c * f))
        if x_mask is None:
            return x, None
        # print(x_mask.size())
        return x, x_mask[:, :, :-2:2][:, :, :-2:2]


class MultiSequential(torch.nn.Sequential):
    """Multi-input multi-output torch.nn.Sequential."""

    def __init__(self, *args, layer_drop_rate=0.0):
        """Initialize MultiSequential with layer_drop.
        Args:
            layer_drop_rate (float): Probability of dropping out each fn (layer).
        """
        super(MultiSequential, self).__init__(*args)
        self.layer_drop_rate = layer_drop_rate

    def forward(self, *args):
        """Repeat."""
        _probs = torch.empty(len(self)).uniform_()
        for idx, m in enumerate(self):
            if not self.training or (_probs[idx] >= self.layer_drop_rate):
                args = m(*args)
        return args
    

def repeat(N, fn, layer_drop_rate=0.0):
    """Repeat module N times.
    Args:
        N (int): Number of repeat time.
        fn (Callable): Function to generate module.
        layer_drop_rate (float): Probability of dropping out each fn (layer).
    Returns:
        MultiSequential: Repeated model instance.
    """
    return MultiSequential(*[fn(n) for n in range(N)], layer_drop_rate=layer_drop_rate)


class PositionwiseFeedForward(torch.nn.Module):
    """Positionwise feed forward layer.
    Args:
        idim (int): Input dimenstion.
        hidden_units (int): The number of hidden units.
        dropout_rate (float): Dropout rate.
    """

    def __init__(self, idim, hidden_units, dropout_rate, activation=torch.nn.ReLU()):
        """Construct an PositionwiseFeedForward object."""
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = torch.nn.Linear(idim, hidden_units)
        self.w_2 = torch.nn.Linear(hidden_units, idim)
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.activation = activation

    def forward(self, x):
        """Forward function."""
        return self.w_2(self.dropout(self.activation(self.w_1(x))))


class transformer_encoder(nn.Module):
    """
    Multimodal Transformer Encoder (Audio + Text) using BERT.
    
    Architecture:
    - Audio Input: (Batch, Time, Features)
        - Processed as Forward and Backward sequences.
        - Subsampling + Positional Encoding + Transformer Encoders.
    - Text Input: Utterance string
        - Tokenized and processed by pre-trained BERT.
        - Hidden states extracted and averaged.
    - Fusion: Concatenation of Audio (Forward+Backward) and Text embeddings.
    - Multi-Encoder: Additional Transformer layers processing the fused representation.
    - Classifier: Linear layers for final prediction.
    
    Args:
        features (int): Input audio feature dimension.
        classes (int): Number of output classes (not explicitly used in constructor, but implied).
        odim (int): Output dimension for BERT and final classifier.
        d_model (int): Dimension of audio embeddings.
        d_model_multi (int): Dimension of fused embeddings.
    """
    def __init__(self, features, classes, kernel_size=2, odim=8, cachedir="./cachedir/", nb_stacks=1, dilations=8, nb_filters=46, d_model=576, dropout_rate=0.15, num_blocks=12, d_model_multi=768, num_blocks_multi=3, n_heads=12, linear_units=4096, normalize_before=True, concat_after=False, stochastic_depth_rate=0.0, add_pos_enc=False):
        super(transformer_encoder, self).__init__()
        self.nb_stacks = nb_stacks
        self.dilations = dilations
        self.kernel_size=kernel_size
        self.nb_filters=nb_filters
        self.pos_enc_bool=add_pos_enc
        self.spatialdropout = nn.Dropout2d(dropout_rate)
        self.maxpool = nn.MaxPool1d(8)
        self.avgpool = nn.AvgPool1d(8)
        self.adaptavgpool = nn.AdaptiveAvgPool1d(10)
        self.bn=nn.BatchNorm1d(32)
        self.gn = nn.GroupNorm(num_groups=1, num_channels=256)
        self.dropout= nn.Dropout(dropout_rate)
        self.dropout2d= nn.Dropout2d(dropout_rate)
        self.sigmoid=nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
        self.flatten=nn.Flatten()
        # self.lstm1 = nn.LSTM(input_size=256, hidden_size=128, num_layers=2, batch_first=True)  
        # self.gru = nn.GRU(256, 128, num_layers=2, bidirectional=True, dropout=dropout_rate, batch_first=True)
        self.subsampling = Conv2dSubsampling(idim=features, odim=d_model, dropout_rate=dropout_rate)    
        encoder_selfattn_layer = MultiHeadedAttention
        sencoder_selfattn_layer = SMultiHeadedAttention

        encoder_selfattn_layer_args = (
            n_heads,
            d_model,
            dropout_rate
        )
        positionwise_layer = PositionwiseFeedForward
        positionwise_layer_args = (d_model, linear_units, dropout_rate)
        self.pos_enc=PositionalEncoding(d_model=d_model, dropout_rate=dropout_rate) 

        self.normalize_before=normalize_before

        self.encoders = repeat(
            num_blocks,
            lambda lnum: EncoderLayer(
                d_model,
                sencoder_selfattn_layer(*encoder_selfattn_layer_args), 
                positionwise_layer(*positionwise_layer_args),
                dropout_rate,
                self.normalize_before,
                concat_after,
                stochastic_depth_rate * float(1 + lnum) / num_blocks,
            ),
        )
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.BERT = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", cache_dir=cachedir, num_labels=odim, output_hidden_states=True, ignore_mismatched_sizes=True)
        
        # self.processor = AutoProcessor.from_pretrained("facebook/hubert-large-ls960-ft")
        # self.model = HubertModel.from_pretrained("facebook/hubert-large-ls960-ft")

        multiencoder_selfattn_layer_args = (
            n_heads,
            d_model_multi,
            dropout_rate
        )

        multipositionwise_layer_args = (d_model_multi, linear_units, dropout_rate)

        self.multiencoders = repeat(
            num_blocks_multi,
            lambda lnum: EncoderLayer(
                d_model_multi,
                sencoder_selfattn_layer(*multiencoder_selfattn_layer_args), 
                positionwise_layer(*multipositionwise_layer_args),
                dropout_rate,
                self.normalize_before,
                concat_after,
                stochastic_depth_rate * float(1 + lnum) / num_blocks,
            ),
        )
        self.classifier2=nn.Sequential(
            nn.Tanh(),
            # nn.AvgPool1d(10),
            # nn.AdaptiveAvgPool1d(10),
            # nn.MaxPool1d(kernel_size=3),
            nn.Flatten(),
            nn.Linear(768, odim)
        )
        self.classifier = nn.Sequential(
            nn.Linear(36,36),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(36, 36),
        )

        self.attention=nn.Sequential(
            nn.Linear(256, 256),
            nn.Dropout(0.2),
            nn.Tanh()
        )
        self.dense=nn.Linear(4248, 7)
        self.after_norm=LayerNorm(d_model)
        self.fc=nn.Linear(576, 768)

        
    
    def forward(self, x, src_mask, utterance):   
        forward = x.float()
        backward = torch.flip(x.float(), dims=[1])
        forward, forward_mask = self.subsampling(forward, src_mask)
        backward, backward_mask = self.subsampling(backward, src_mask)

        if self.pos_enc_bool==True:
            forward=self.pos_enc(forward)
            backward=self.pos_enc(backward)

        forward, _ = self.encoders(forward, forward_mask)
        backward, _ = self.encoders(backward, backward_mask)

        if self.normalize_before==True:
            forward=self.after_norm(forward)
            backward=self.after_norm(backward)
            # forward=self.avgpool(forward)
            # backward=self.avgpool(backward)
            # print(forward.shape)
        # forward, _ = torch.max(forward, dim=2) # probiere torch.mean und flatten # global average pooling ausprobieren
        # backward, _ = torch.max(backward, dim=2)
        # print(forward_mask.shape, backward_mask.shape)
        x_mask = torch.cat([forward_mask, backward_mask], axis=2)
        x = torch.cat([forward, backward], axis=1)
        # print(x.shape)
        inputs = self.tokenizer(utterance, return_tensors='pt', truncation=True, padding='longest', max_length=512)
        attention_mask = inputs['attention_mask']
        # print(attention_mask.shape)
        outputs = self.BERT(**inputs)
        # print(type(outputs))
        hidden_states=outputs.hidden_states[-4:] 
        # print(hidden_states[0].shape)
        hidden_states=sum(hidden_states)/len(hidden_states)
        # print(hidden_states.shape)
        x = self.fc(x)
        # print(x_mask.shape)
        attention_mask=attention_mask.unsqueeze(1)
        x_mask=torch.cat([x_mask, attention_mask], dim=2)
        # print(x_mask.shape)
        x = torch.cat([x, hidden_states], dim=1)
        # print(x.shape)
        x, _ = self.multiencoders(x, x_mask)
        # print(x.shape)
        x, _ = torch.max(x, dim=1)
        x = self.classifier2(x)
        # print(x.shape)
        return x