import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (seq_len, batch_size, d_model)
        return x + self.pe[:x.size(0), :]

class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism"""
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(1)
        seq_len = query.size(0)
        
        # Linear transformations
        Q = self.W_q(query)  # (seq_len, batch_size, d_model)
        K = self.W_k(key)
        V = self.W_v(value)
        
        # Reshape for multi-head attention
        Q = Q.view(seq_len, batch_size, self.num_heads, self.d_k).transpose(0, 2)  # (num_heads, batch_size, seq_len, d_k)
        K = K.view(seq_len, batch_size, self.num_heads, self.d_k).transpose(0, 2)
        V = V.view(seq_len, batch_size, self.num_heads, self.d_k).transpose(0, 2)
        
        # Transpose for matrix multiplication
        Q = Q.transpose(1, 2)  # (num_heads, seq_len, batch_size, d_k)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        
        # Attention calculation
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        context = torch.matmul(attn_weights, V)
        
        # Reshape back
        context = context.transpose(1, 2)  # (num_heads, batch_size, seq_len, d_k)
        context = context.transpose(0, 2)  # (seq_len, batch_size, num_heads, d_k)
        context = context.contiguous().view(seq_len, batch_size, self.d_model)
        
        # Final linear transformation
        output = self.W_o(context)
        
        return output

class TransformerEncoderLayer(nn.Module):
    """Single transformer encoder layer"""
    def __init__(self, d_model, num_heads, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, src_mask=None):
        # Self-attention
        src2 = self.self_attn(src, src, src, src_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        
        # Feedforward
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        
        return src

class TransformerDecoderLayer(nn.Module):
    """Single transformer decoder layer"""
    def __init__(self, d_model, num_heads, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        # Self-attention on target
        tgt2 = self.self_attn(tgt, tgt, tgt, tgt_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        
        # Cross-attention with encoder output
        tgt2 = self.cross_attn(tgt, memory, memory, memory_mask)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        
        # Feedforward
        tgt2 = self.linear2(self.dropout(F.relu(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        
        return tgt

class TimeSeriesTransformer(nn.Module):
    """Complete Encoder-Decoder Transformer for Time Series"""
    def __init__(self, input_size, output_size, d_model=128, num_heads=8, 
                 num_encoder_layers=3, num_decoder_layers=3, 
                 dim_feedforward=512, dropout=0.1, seq_len=30, pred_len=1):
        super().__init__()
        
        self.input_size = input_size
        self.output_size = output_size
        self.d_model = d_model
        self.seq_len = seq_len
        self.pred_len = pred_len
        
        # Input/Output projections
        self.input_projection = nn.Linear(input_size, d_model)
        self.output_projection = nn.Linear(d_model, output_size)
        
        # Target embedding for decoder
        self.target_embedding = nn.Linear(output_size, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len=seq_len + pred_len)
        
        # Encoder layers
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, dim_feedforward, dropout)
            for _ in range(num_encoder_layers)
        ])
        
        # Decoder layers
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, num_heads, dim_feedforward, dropout)
            for _ in range(num_decoder_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        
        # Initialize parameters
        self.init_weights()
    
    def init_weights(self):
        """Initialize model weights"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def generate_square_subsequent_mask(self, sz):
        """Generate causal mask for decoder"""
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask
    
    def encode(self, src):
        """Encoder forward pass"""
        # src: (batch_size, seq_len, input_size)
        batch_size, seq_len, _ = src.shape
        
        # Project input to model dimension and transpose
        src = self.input_projection(src)  # (batch_size, seq_len, d_model)
        src = src.transpose(0, 1)  # (seq_len, batch_size, d_model)
        
        # Add positional encoding
        src = self.pos_encoder(src)
        src = self.dropout(src)
        
        # Pass through encoder layers
        for layer in self.encoder_layers:
            src = layer(src)
        
        return src  # (seq_len, batch_size, d_model)
    
    def decode(self, tgt, memory, tgt_mask=None):
        """Decoder forward pass"""
        # tgt: (batch_size, tgt_len, output_size)
        # memory: (seq_len, batch_size, d_model)
        
        batch_size, tgt_len, _ = tgt.shape
        
        # Project target to model dimension and transpose
        tgt = self.target_embedding(tgt)  # (batch_size, tgt_len, d_model)
        tgt = tgt.transpose(0, 1)  # (tgt_len, batch_size, d_model)
        
        # Add positional encoding
        tgt = self.pos_encoder(tgt)
        tgt = self.dropout(tgt)
        
        # Pass through decoder layers
        for layer in self.decoder_layers:
            tgt = layer(tgt, memory, tgt_mask)
        
        # Project back to output dimension
        tgt = tgt.transpose(0, 1)  # (batch_size, tgt_len, d_model)
        output = self.output_projection(tgt)  # (batch_size, tgt_len, output_size)
        
        return output
    
    def forward(self, src, tgt=None):
        """Forward pass"""
        # src: (batch_size, seq_len, input_size)
        batch_size = src.size(0)
        device = src.device
        
        # Encode input sequence
        memory = self.encode(src)  # (seq_len, batch_size, d_model)
        
        if self.training and tgt is not None:
            # Training mode with teacher forcing
            # tgt: (batch_size, pred_len, output_size)
            tgt_len = tgt.size(1)
            tgt_mask = self.generate_square_subsequent_mask(tgt_len).to(device)
            
            # Create decoder input by shifting target right and padding with zeros
            decoder_input = torch.zeros(batch_size, tgt_len, self.output_size, device=device)
            if tgt_len > 1:
                decoder_input[:, 1:] = tgt[:, :-1]
            
            # Decode
            output = self.decode(decoder_input, memory, tgt_mask)
            return output
        else:
            # Inference mode - autoregressive generation
            outputs = []
            
            # Start with zeros
            current_input = torch.zeros(batch_size, 1, self.output_size, device=device)
            
            for _ in range(self.pred_len):
                # Generate mask for current sequence
                tgt_len = current_input.size(1)
                tgt_mask = self.generate_square_subsequent_mask(tgt_len).to(device)
                
                # Decode current sequence
                output = self.decode(current_input, memory, tgt_mask)
                
                # Take the last prediction
                next_pred = output[:, -1:, :]  # (batch_size, 1, output_size)
                outputs.append(next_pred)
                
                # Append to input for next iteration
                current_input = torch.cat([current_input, next_pred], dim=1)
            
            # Concatenate all predictions
            return torch.cat(outputs, dim=1)  # (batch_size, pred_len, output_size)

# Simple version for step-by-step prediction (like LSTM)
class SimpleTransformerForecast(nn.Module):
    """Simplified Transformer for LSTM-like step-by-step forecasting"""
    def __init__(self, input_size, d_model=128, num_heads=8, num_layers=3, 
                 dim_feedforward=512, dropout=0.1, seq_len=30):
        super().__init__()
        
        self.input_size = input_size
        self.d_model = d_model
        self.seq_len = seq_len
        
        # Input projection
        self.input_projection = nn.Linear(input_size, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len=seq_len)
        
        # Transformer encoder layers
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        
        # Output projection (like LSTM fc layer)
        self.output_projection = nn.Linear(d_model, input_size)
        
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self.init_weights()
    
    def init_weights(self):
        """Initialize model weights"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x):
        """Forward pass - similar to LSTM"""
        # x: (batch_size, seq_len, input_size)
        batch_size, seq_len, _ = x.shape
        
        # Project input to model dimension
        x = self.input_projection(x)  # (batch_size, seq_len, d_model)
        x = x.transpose(0, 1)  # (seq_len, batch_size, d_model)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        x = self.dropout(x)
        
        # Pass through encoder layers
        for layer in self.encoder_layers:
            x = layer(x)
        
        # Transpose back and project to output
        x = x.transpose(0, 1)  # (batch_size, seq_len, d_model)
        output = self.output_projection(x)  # (batch_size, seq_len, input_size)
        
        return output
