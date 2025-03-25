import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PatchEmbedding(nn.Module):
    """
    Split image into patches and embed them
    """
    def __init__(self, img_size=224, patch_size=16, in_channels=1, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        
        # Create projection layer to convert patches to embeddings
        self.proj = nn.Conv2d(
            in_channels, 
            embed_dim, 
            kernel_size=patch_size, 
            stride=patch_size
        )
    
    def forward(self, x):
        """
        Forward pass
        x: (batch_size, in_channels, img_size, img_size)
        """
        # (batch_size, embed_dim, n_patches^0.5, n_patches^0.5)
        x = self.proj(x)
        # (batch_size, embed_dim, n_patches)
        x = x.flatten(2)
        # (batch_size, n_patches, embed_dim)
        x = x.transpose(1, 2)
        return x

class MultiHeadAttention(nn.Module):
    """
    Multi-head attention mechanism
    """
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        # Linear projections
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Forward pass
        x: (batch_size, seq_len, embed_dim)
        """
        batch_size, seq_len, _ = x.size()
        
        # Linear projection and reshape to separate heads
        # (batch_size, seq_len, embed_dim * 3) -> (batch_size, seq_len, 3, num_heads, head_dim)
        qkv = self.qkv(x).reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        # (3, batch_size, num_heads, seq_len, head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        
        # Separate query, key, value
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Compute scaled dot-product attention
        # (batch_size, num_heads, seq_len, head_dim) @ (batch_size, num_heads, head_dim, seq_len)
        # -> (batch_size, num_heads, seq_len, seq_len)
        attn = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Multiply by values
        # (batch_size, num_heads, seq_len, seq_len) @ (batch_size, num_heads, seq_len, head_dim)
        # -> (batch_size, num_heads, seq_len, head_dim)
        out = attn @ v
        
        # Reshape back to original dimensions
        # (batch_size, num_heads, seq_len, head_dim) -> (batch_size, seq_len, num_heads, head_dim)
        out = out.transpose(1, 2)
        # (batch_size, seq_len, num_heads, head_dim) -> (batch_size, seq_len, embed_dim)
        out = out.reshape(batch_size, seq_len, self.embed_dim)
        
        # Output projection
        out = self.proj(out)
        out = self.dropout(out)
        
        return out

class MLP(nn.Module):
    """
    Multi-layer perceptron
    """
    def __init__(self, embed_dim, mlp_dim, dropout=0.0):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class TransformerBlock(nn.Module):
    """
    Transformer encoder block
    """
    def __init__(self, embed_dim, num_heads, mlp_dim, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, mlp_dim, dropout)
        
    def forward(self, x):
        # Self-attention with residual connection
        x = x + self.attn(self.norm1(x))
        # MLP with residual connection
        x = x + self.mlp(self.norm2(x))
        return x

class VisionTransformer(nn.Module):
    """
    Vision Transformer (ViT) for image classification
    """
    def __init__(
        self, 
        img_size=224, 
        patch_size=16, 
        in_channels=1, 
        num_classes=2,
        embed_dim=768, 
        depth=12, 
        num_heads=12, 
        mlp_dim=3072, 
        dropout=0.1
    ):
        super().__init__()
        
        # Image patch embedding
        self.patch_embed = PatchEmbedding(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim
        )
        
        # Number of patches
        self.n_patches = self.patch_embed.n_patches
        
        # Class token and position embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_patches + 1, embed_dim))
        self.dropout = nn.Dropout(dropout)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_dim, dropout)
            for _ in range(depth)
        ])
        
        # Layer normalization
        self.norm = nn.LayerNorm(embed_dim)
        
        # Classification head
        self.head = nn.Linear(embed_dim, num_classes)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        # Initialize patch embedding
        nn.init.normal_(self.patch_embed.proj.weight, std=0.02)
        
        # Initialize class token
        nn.init.normal_(self.cls_token, std=0.02)
        
        # Initialize position embedding
        nn.init.normal_(self.pos_embed, std=0.02)
        
        # Initialize MLP weights
        for block in self.blocks:
            nn.init.normal_(block.mlp.fc1.weight, std=0.02)
            nn.init.normal_(block.mlp.fc2.weight, std=0.02)
            
        # Initialize attention projection
        for block in self.blocks:
            nn.init.normal_(block.attn.qkv.weight, std=0.02)
            nn.init.normal_(block.attn.proj.weight, std=0.02)
            
        # Initialize classification head
        nn.init.normal_(self.head.weight, std=0.02)
        
    def forward(self, x):
        # Get batch size
        batch_size = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)
        
        # Add class token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Add position embedding
        x = x + self.pos_embed
        x = self.dropout(x)
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Layer normalization
        x = self.norm(x)
        
        # Use class token for classification
        x = x[:, 0]
        
        # Classification head
        x = self.head(x)
        
        return x

# Define a smaller ViT for faster training
class SmallViT(nn.Module):
    """
    A smaller Vision Transformer for faster training
    """
    def __init__(
        self, 
        img_size=224, 
        patch_size=32, 
        in_channels=1, 
        num_classes=2,
        embed_dim=192, 
        depth=4, 
        num_heads=3, 
        mlp_dim=768, 
        dropout=0.1
    ):
        super().__init__()
        
        # Create full ViT with smaller parameters
        self.vit = VisionTransformer(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            num_classes=num_classes,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_dim=mlp_dim,
            dropout=dropout
        )
        
    def forward(self, x):
        return self.vit(x) 