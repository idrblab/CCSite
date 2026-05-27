"""
Main model architecture for covalent cysteine site prediction.

This module contains the core neural network components including:
- SelfAttention: Multi-head attention mechanism
- Encoder: Convolutional protein feature extractor  
- Decoder: Transformer decoder with attention
- Predictor: Complete model combining encoder-decoder
- Trainer/Tester: Training and evaluation classes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Tuple, Optional, List

from src.utils.helpers import prepare_batch_embeddings
from peft import get_peft_model_state_dict

from src.models.radam import RAdam


class FocalLoss(nn.Module):
    """Binary focal loss that accepts a scalar positive-class weight."""

    def __init__(self, alpha: Optional[float] = None, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        if alpha is not None:
            alpha_tensor = torch.tensor(float(alpha), dtype=torch.float32)
            self.register_buffer("alpha", alpha_tensor)
        else:
            self.alpha = None
        self.gamma = gamma
        if reduction not in {"none", "mean", "sum"}:
            raise ValueError("reduction must be 'none', 'mean', or 'sum'")
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.long().view(-1)
        log_probs = F.log_softmax(logits, dim=1)
        probs = torch.exp(log_probs)
        log_pt = log_probs.gather(1, targets.unsqueeze(1)).view(-1)
        pt = probs.gather(1, targets.unsqueeze(1)).view(-1)

        if hasattr(self, "alpha") and self.alpha is not None:
            if logits.size(1) != 2:
                raise ValueError("Scalar alpha is only supported for binary classification")
            alpha_pos = self.alpha
            alpha_neg = 1.0 - self.alpha
            alpha_t = torch.where(targets == 1, alpha_pos, alpha_neg)
        else:
            alpha_t = torch.ones_like(pt)

        focal_factor = (1 - pt).pow(self.gamma)
        loss = -alpha_t * focal_factor * log_pt

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class SelfAttention(nn.Module):
    """Multi-head self-attention mechanism."""
    
    def __init__(self, hid_dim: int, n_heads: int, dropout: float, device: torch.device):
        super().__init__()
        self.hid_dim = hid_dim
        self.n_heads = n_heads
        
        assert hid_dim % n_heads == 0, "Hidden dimension must be divisible by number of heads"
        
        self.w_q = nn.Linear(hid_dim, hid_dim)
        self.w_k = nn.Linear(hid_dim, hid_dim)
        self.w_v = nn.Linear(hid_dim, hid_dim)
        self.fc = nn.Linear(hid_dim, hid_dim)
        self.do = nn.Dropout(dropout)
        self.register_buffer("scale", torch.sqrt(torch.FloatTensor([hid_dim // n_heads])))
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask_q: Optional[torch.Tensor] = None,
        mask_k: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of self-attention.
        
        Args:
            query: Query tensor [batch_size, seq_len, hid_dim]
            key: Key tensor [batch_size, seq_len, hid_dim]
            value: Value tensor [batch_size, seq_len, hid_dim]
            mask_q: Optional query mask [batch_size, query_len]
            mask_k: Optional key mask [batch_size, key_len]
            
        Returns:
            output: Attention output [batch_size, seq_len, hid_dim]
            attention: Attention weights [batch_size, n_heads, seq_len, seq_len]
        """
        batch_size = query.shape[0]
        
        # Linear transformations
        Q = self.w_q(query)
        K = self.w_k(key)
        V = self.w_v(value)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)
        
        # Scaled dot-product attention
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale

        # Support legacy single-mask usage
        if mask_k is None and mask_q is not None and mask_q.dim() == 4:
            mask_k = mask_q
            mask_q = None

        mask_k_exp = None
        if mask_k is not None:
            mask_k_exp = mask_k
            if mask_k_exp.dim() == 2:
                mask_k_exp = mask_k_exp.unsqueeze(1).unsqueeze(2)
            elif mask_k_exp.dim() == 3:
                mask_k_exp = mask_k_exp.unsqueeze(1)
            energy = energy.masked_fill(~mask_k_exp, -1e10)

        mask_q_exp = None
        if mask_q is not None:
            mask_q_exp = mask_q
            if mask_q_exp.dim() == 2:
                mask_q_exp = mask_q_exp.unsqueeze(1).unsqueeze(-1)
            elif mask_q_exp.dim() == 3:
                mask_q_exp = mask_q_exp.unsqueeze(-1)
            energy = energy.masked_fill(~mask_q_exp, -1e10)

        attention = F.softmax(energy, dim=-1)
        if mask_q_exp is not None:
            attention = attention.masked_fill(~mask_q_exp, 0.0)
        if mask_k_exp is not None:
            attention = attention.masked_fill(~mask_k_exp, 0.0)
        attention = self.do(attention)
        x = torch.matmul(attention, V)
        
        # Reshape back
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(batch_size, -1, self.n_heads * (self.hid_dim // self.n_heads))
        if mask_q is not None:
            q_mask_float = mask_q.float().unsqueeze(-1)
            x = x * q_mask_float
        x = self.fc(x)
        
        return x, attention


class Encoder(nn.Module):
    """Convolutional encoder for protein feature extraction."""
    
    def __init__(self, protein_dim: int, hid_dim: int, n_layers: int, 
                 kernel_size: int, dropout: float, device: torch.device):
        super().__init__()
        
        assert kernel_size % 2 == 1, "Kernel size must be odd"
        
        self.input_dim = protein_dim
        self.hid_dim = hid_dim
        self.kernel_size = kernel_size
        self.n_layers = n_layers
        self.device = device
        
        self.register_buffer("scale", torch.sqrt(torch.FloatTensor([0.5])))
        self.convs = nn.ModuleList([
            nn.Conv1d(hid_dim, 2 * hid_dim, kernel_size, padding=(kernel_size - 1) // 2)
            for _ in range(n_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(protein_dim, hid_dim)
        self.ln = nn.LayerNorm(hid_dim)
    
    def forward(self, protein: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of encoder.
        
        Args:
            protein: Protein embeddings [batch_size, seq_len, protein_dim]
            
        Returns:
            Encoded features [batch_size, seq_len, hid_dim]
        """
        # Linear projection
        conv_input = self.fc(protein)
        conv_input = conv_input.permute(0, 2, 1)  # [batch_size, hid_dim, seq_len]
        
        # Convolutional layers with GLU activation
        for conv in self.convs:
            conved = conv(self.dropout(conv_input))
            conved = F.glu(conved, dim=1)  # Gated Linear Unit
            conved = (conved + conv_input) * self.scale  # Residual connection
            conv_input = conved
        
        # Back to original shape and normalize
        conved = conved.permute(0, 2, 1)
        conved = self.ln(conved)
        
        return conved


class PositionwiseFeedforward(nn.Module):
    """Position-wise feedforward network."""
    
    def __init__(self, hid_dim: int, pf_dim: int, dropout: float):
        super().__init__()
        self.fc_1 = nn.Conv1d(hid_dim, pf_dim, 1)
        self.fc_2 = nn.Conv1d(pf_dim, hid_dim, 1)
        self.do = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of feedforward network."""
        x = x.permute(0, 2, 1)
        x = self.do(F.relu(self.fc_1(x)))
        x = self.fc_2(x)
        x = x.permute(0, 2, 1)
        return x


class DecoderLayer(nn.Module):
    """Single decoder layer with self-attention and cross-attention."""
    
    def __init__(self, hid_dim: int, n_heads: int, pf_dim: int, 
                 self_attention, positionwise_feedforward, dropout: float, device: torch.device):
        super().__init__()
        self.ln = nn.LayerNorm(hid_dim)
        self.sa = self_attention(hid_dim, n_heads, dropout, device)  # Self-attention
        self.ea = self_attention(hid_dim, n_heads, dropout, device)  # Cross-attention
        self.pf = positionwise_feedforward(hid_dim, pf_dim, dropout)
        self.do = nn.Dropout(dropout)
    
    def forward(self, trg: torch.Tensor, src: torch.Tensor, 
                trg_mask: Optional[torch.Tensor] = None, 
                src_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of decoder layer.
        
        Args:
            trg: Target sequence [batch_size, tgt_len, hid_dim]
            src: Source sequence [batch_size, src_len, hid_dim]
            trg_mask: Target mask
            src_mask: Source mask
            
        Returns:
            output: Layer output [batch_size, tgt_len, hid_dim]
            attention: Cross-attention weights
        """
        # Self-attention
        trg_1 = trg
        trg_mask_bool = trg_mask if trg_mask is not None else None
        trg, _ = self.sa(trg, trg, trg, trg_mask_bool, trg_mask_bool)
        trg = self.ln(trg_1 + self.do(trg))
        
        # Cross-attention
        trg_2 = trg
        src_mask_bool = src_mask if src_mask is not None else None
        trg, attention = self.ea(trg, src, src, trg_mask_bool, src_mask_bool)
        trg = self.ln(trg_2 + self.do(trg))
        
        # Feedforward
        trg_3 = trg
        trg = self.ln(trg_3 + self.do(self.pf(trg)))
        
        return trg, attention


class Decoder(nn.Module):
    """Transformer decoder for covalent cysteine-site prediction."""
    
    def __init__(self, local_dim: int, hid_dim: int, n_layers: int, n_heads: int, 
                 pf_dim: int, decoder_layer, self_attention, positionwise_feedforward, 
                 dropout: float, device: torch.device):
        super().__init__()
        self.hid_dim = hid_dim
        self.device = device
        
        self.layers = nn.ModuleList([
            decoder_layer(hid_dim, n_heads, pf_dim, self_attention, 
                         positionwise_feedforward, dropout, device)
            for _ in range(n_layers)
        ])
        
        self.ft = nn.Linear(local_dim, hid_dim)
        self.do = nn.Dropout(dropout)

        # Learnable pooling query for attention-style aggregation
        self.pool_query = nn.Parameter(torch.randn(hid_dim))
        nn.init.normal_(self.pool_query, std=1.0 / math.sqrt(hid_dim))
        
        # Classification head
        self.fc_1 = nn.Linear(hid_dim, 256)
        self.norm_1 = nn.LayerNorm(256)
        self.fc_2 = nn.Linear(256, 32)
        self.norm_2 = nn.LayerNorm(32)
        self.fc_3 = nn.Linear(32, 2)
        self.head_dropout = nn.Dropout(dropout)
        self.logit_temperature = nn.Parameter(torch.tensor(1.0))
    
    def forward(self, trg: torch.Tensor, src: torch.Tensor, 
                trg_mask: Optional[torch.Tensor] = None, 
                src_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of decoder.
        
        Args:
            trg: Local features [batch_size, local_len, local_dim]
            src: Encoded protein features [batch_size, protein_len, hid_dim]
            trg_mask: Target mask
            src_mask: Source mask
            
        Returns:
            sum_features: Pooled features [batch_size, hid_dim]
            attention: Final attention weights
            logits: Classification logits [batch_size, 2]
        """
        # Project local features to hidden dimension
        trg = self.ft(trg)
        
        # Pass through decoder layers
        for layer in self.layers:
            trg, attention = layer(trg, src, trg_mask, src_mask)
        
        # Attention-style pooling with learnable query
        pooling_mask = trg_mask if trg_mask is not None else None

        scores = torch.matmul(trg, self.pool_query)
        scores = scores / math.sqrt(self.hid_dim)
        if pooling_mask is not None:
            scores = scores.masked_fill(~pooling_mask, float("-inf"))

        norm = F.softmax(scores, dim=1)
        if pooling_mask is not None:
            norm = norm.masked_fill(~pooling_mask, 0.0)

        sum_features = torch.sum(trg * norm.unsqueeze(-1), dim=1)
        
        # Classification head
        logits = self.fc_1(sum_features)
        logits = self.norm_1(logits)
        logits = F.gelu(logits)
        logits = self.head_dropout(logits)

        logits = self.fc_2(logits)
        logits = self.norm_2(logits)
        logits = F.gelu(logits)
        logits = self.head_dropout(logits)

        logits = self.fc_3(logits)
        temperature = F.softplus(self.logit_temperature) + 1e-6
        logits = logits / temperature
        
        return sum_features, attention, logits


class Predictor(nn.Module):
    """Complete covalent cysteine-site predictor model."""
    
    def __init__(self, encoder: Encoder, decoder: Decoder, device: torch.device, focal_alpha: float = 0.5):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.focal_alpha = float(focal_alpha)
        self.loss_fn = FocalLoss(alpha=self.focal_alpha, gamma=2.0, reduction="mean").to(device)
    
    def make_masks(self, local_num: List[int], protein_num: List[int], 
                   local_max_len: int, protein_max_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create attention masks for variable-length sequences."""
        N = len(local_num)
        local_mask = torch.zeros((N, local_max_len), dtype=torch.bool, device=self.device)
        protein_mask = torch.zeros((N, protein_max_len), dtype=torch.bool, device=self.device)
        
        for i in range(N):
            local_mask[i, :local_num[i]] = True
            protein_mask[i, :protein_num[i]] = True
        
        return local_mask, protein_mask
    
    def forward(self, local: torch.Tensor, protein: torch.Tensor, 
                local_num: List[int], protein_num: List[int]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of complete model.
        
        Args:
            local: Local features [batch_size, local_max_len, local_dim]
            protein: Protein features [batch_size, protein_max_len, protein_dim]
            local_num: Actual lengths of local sequences
            protein_num: Actual lengths of protein sequences
            
        Returns:
            sum_features: Pooled features
            attention: Attention weights
            logits: Classification logits
        """
        local_max_len = local.shape[1]
        protein_max_len = protein.shape[1]
        local_mask, protein_mask = self.make_masks(local_num, protein_num, local_max_len, protein_max_len)
        
        # Encode protein features
        enc_src = self.encoder(protein)
        
        # Decode and predict
        sum_features, attention, logits = self.decoder(local, enc_src, local_mask, protein_mask)
        
        return sum_features, attention, logits
    
    def train_step(self, data: Tuple):
        """Compute training loss from a data pack."""
        local, protein, correct_interaction, local_num, protein_num = data
        sum_features, attention, predicted_interaction = self.forward(local, protein, local_num, protein_num)
        del sum_features, attention  # Memory cleanup
        loss = self.loss_fn(predicted_interaction, correct_interaction)
        return loss

    def predict_batch(self, data: Tuple, compute_loss: bool = False):
        """Run a forward pass for evaluation and optionally compute loss."""
        local, protein, correct_interaction, local_num, protein_num = data

        sum_features, attention, predicted_interaction = self.forward(local, protein, local_num, protein_num)
        del sum_features, attention  # Memory cleanup

        probs = F.softmax(predicted_interaction, dim=1)
        probs_cpu = probs.detach().to('cpu').numpy()
        predicted_labels = np.argmax(probs_cpu, axis=1)
        predicted_scores = probs_cpu[:, 1]
        correct_labels = correct_interaction.detach().to('cpu').numpy()

        if compute_loss:
            loss = self.loss_fn(predicted_interaction, correct_interaction)
            return loss, correct_labels, predicted_labels, predicted_scores

        return correct_labels, predicted_labels, predicted_scores


class Trainer:
    """Training class for the covalent cysteine-site predictor."""

    def __init__(
        self,
        model: Predictor,
        lr: float,
        weight_decay: float,
        embedding_model: nn.Module,
        window_size: int,
        embedding_lr: Optional[float] = None,
        lora_enabled: bool = True,
        initialize_model: bool = True,
    ):
        self.model = model
        self.embedding_model = embedding_model
        self.window_size = int(window_size)
        self.embedding_lr = embedding_lr if embedding_lr is not None else lr
        self.lora_enabled = bool(lora_enabled)
        self.initialize_model = bool(initialize_model)
        
        # Separate weight and bias parameters for different regularization
        weight_p, bias_p = [], []
        
        # Initialize model weights only for fresh training runs.
        # When resuming from a checkpoint, keep loaded weights intact.
        if self.initialize_model:
            for p in self.model.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
        
        # Group parameters
        for name, p in self.model.named_parameters():
            if 'bias' in name:
                bias_p += [p]
            else:
                weight_p += [p]
        
        optimizer_groups = [
            {'params': weight_p, 'weight_decay': weight_decay, 'lr': lr},
            {'params': bias_p, 'weight_decay': 0.0, 'lr': lr},
        ]

        embedding_params = [p for p in self.embedding_model.parameters() if p.requires_grad]
        if embedding_params:
            optimizer_groups.append({
                'params': embedding_params,
                'weight_decay': 0.0,
                'lr': self.embedding_lr,
            })

        self.optimizer = RAdam(optimizer_groups, lr=lr)

        if not self.lora_enabled:
            self.embedding_model.eval()

        # self.optimizer = Lookahead(self.optimizer, k=5, alpha=0.5)
    
    def train(self, dataloader, device: torch.device) -> None:
        """Train the model for one epoch and update parameters only."""
        self.model.train()
        if self.lora_enabled:
            self.embedding_model.train()
        else:
            self.embedding_model.eval()
        
        for batch_idx, batch in enumerate(dataloader):

            local, protein, label_tensor, local_num, protein_num = prepare_batch_embeddings(
                self.embedding_model,
                batch["sequences"],
                batch["positions"],
                batch["labels"],
                batch["protein_indices"],
                self.window_size,
                device,
                requires_grad=self.lora_enabled,
            )
            data_pack = (local, protein, label_tensor, local_num, protein_num)
            
            # Clear gradients
            self.optimizer.zero_grad()
            
            # Forward pass and loss calculation
            loss = self.model.train_step(data_pack)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            del data_pack


class Tester:
    """Testing/evaluation class for the covalent cysteine-site predictor."""

    def __init__(self, model: Predictor, embedding_model: nn.Module, window_size: int):
        self.model = model
        self.embedding_model = embedding_model
        self.window_size = int(window_size)

    def test(self, dataloader, device: torch.device, return_loss: bool = False):
        """Evaluate the model on a given dataloader."""
        self.model.eval()
        self.embedding_model.eval()
        T, Y, S = [], [], []
        loss_total = 0.0
        sample_count = 0
        metadata: List[dict] = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):

                local, protein, label_tensor, local_num, protein_num = prepare_batch_embeddings(
                    self.embedding_model,
                    batch["sequences"],
                    batch["positions"],
                    batch["labels"],
                    batch["protein_indices"],
                    self.window_size,
                    device,
                    requires_grad=False,
                )
                data_pack = (local, protein, label_tensor, local_num, protein_num)
                
                if return_loss:
                    loss, correct_labels, predicted_labels, predicted_scores = self.model.predict_batch(
                        data_pack, compute_loss=True
                    )
                    batch_size = len(correct_labels)
                    loss_total += loss.item() * batch_size
                    sample_count += batch_size
                else:
                    correct_labels, predicted_labels, predicted_scores = self.model.predict_batch(
                        data_pack, compute_loss=False
                    )
                
                T.extend(correct_labels)
                Y.extend(predicted_labels)
                S.extend(predicted_scores)

                names = batch.get("names", [""] * len(correct_labels))
                positions = batch["positions"].tolist()
                for name, pos in zip(names, positions):
                    metadata.append({"name": name, "position": int(pos)})
                
                del data_pack
        
        if return_loss:
            return T, Y, S, loss_total, sample_count, metadata
        
        return T, Y, S, metadata
    
    def save_AUCs(self, AUCs: List, filename: str):
        """Save evaluation metrics to file."""
        with open(filename, 'a') as f:
            f.write('\t'.join(map(str, AUCs)) + '\n')
    
    def save_model(self, model: nn.Module, filename: str):
        """Save model state dict."""
        if hasattr(model, 'module'):
            predictor_state = model.module.state_dict()
        else:
            predictor_state = model.state_dict()

        embedding_state = None
        if hasattr(self.embedding_model, "peft_config"):
            try:
                embedding_state = get_peft_model_state_dict(self.embedding_model)
            except Exception:
                embedding_state = self.embedding_model.state_dict()

        payload = {
            'predictor': predictor_state,
            'embedding': embedding_state,
        }
        torch.save(payload, filename)
