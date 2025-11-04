import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


class VAE(nn.Module):
    def __init__(self, input_dim=128, latent_dim=32, d_model=256, nhead=4, num_layers=4, dropout=0.1):
        super(VAE, self).__init__()
        self.input_dim = input_dim  # 输入向量维度
        self.latent_dim = latent_dim  # 隐空间维度
        self.d_model = d_model  # Transformer模型维度
        
        # 编码器输入投影
        self.encoder_input_proj = nn.Linear(input_dim, d_model)
        self.encoder_pos_encoding = nn.Parameter(torch.zeros(1, 1, d_model))  # 固定位置编码
        
        # Transformer编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # VAE核心：投影到均值和方差
        self.fc_mu = nn.Linear(d_model, latent_dim)
        self.fc_logvar = nn.Linear(d_model, latent_dim)
        
        # 解码器输入投影
        self.decoder_input_proj = nn.Linear(latent_dim, d_model)
        self.decoder_pos_encoding = nn.Parameter(torch.zeros(1, 1, d_model))  # 固定位置编码
        
        # Transformer解码器层
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=num_layers)
        
        # 输出投影层
        self.output_projection = nn.Linear(d_model, input_dim)
    
    def encode(self, x):
        """编码到隐空间，返回均值和方差"""
        # 确保输入被正确补长或截断到input_dim
        if x.size(1) < self.input_dim:
            x = F.pad(x, (0, self.input_dim - x.size(1)), 'constant', 0)
        elif x.size(1) > self.input_dim:
            x = x[:, :self.input_dim]
            
        batch_size = x.shape[0]
        
        # 输入投影和位置编码
        x_encoded = self.encoder_input_proj(x)  # [batch_size, d_model]
        x_encoded = x_encoded.unsqueeze(1)  # [batch_size, 1, d_model]
        x_encoded = x_encoded + self.encoder_pos_encoding  # 添加位置编码
        
        # Transformer编码
        encoded = self.encoder(x_encoded)  # [batch_size, 1, d_model]
        encoded_flat = encoded.squeeze(1)  # [batch_size, d_model]
        
        # 投影到隐空间的均值和方差
        mu = self.fc_mu(encoded_flat)  # [batch_size, latent_dim]
        logvar = self.fc_logvar(encoded_flat)  # [batch_size, latent_dim]
        
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """重参数化技巧，使得可以从隐变量分布中采样并保持可微分性"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)  # 从标准正态分布采样
        z = mu + eps * std  # 重参数化采样
        return z
    
    def decode(self, z):
        """从隐变量解码回原始空间"""
        # 从隐空间投影到解码空间
        z_proj = self.decoder_input_proj(z)  # [batch_size, d_model]
        z_proj = z_proj.unsqueeze(1)  # [batch_size, 1, d_model]
        z_proj = z_proj + self.decoder_pos_encoding  # 添加位置编码
        
        # Transformer解码
        decoded = self.decoder(z_proj)  # [batch_size, 1, d_model]
        decoded_flat = decoded.squeeze(1)  # [batch_size, d_model]
        
        # 输出投影
        output = self.output_projection(decoded_flat)  # [batch_size, input_dim]
        
        return output
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入向量 [batch_size, actual_dim]
        Returns:
            recon_x: 重建向量 [batch_size, input_dim]
            mu: 隐变量均值 [batch_size, latent_dim]
            logvar: 隐变量对数方差 [batch_size, latent_dim]
            z: 采样的隐变量 [batch_size, latent_dim]
        """
        # 编码、重参数化、解码
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        
        return recon_x, mu, logvar, z

    def get_latent(self, x):
        """获取输入的隐变量表示（用于降维）"""
        with torch.no_grad():
            mu, _ = self.encode(x)
            return mu

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

# VAE训练函数
def train_vae(vae, train_loader, val_loader, optimizer, device, num_epochs=50, kl_weight=1.0):
    """
    训练VAE模型
    
    Args:
        vae: VAE模型实例
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        optimizer: 优化器
        device: 运行设备
        num_epochs: 训练轮数
        kl_weight: KL散度损失的权重
    
    Returns:
        train_losses: 训练损失列表
        val_losses: 验证损失列表
    """
    vae.train()
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    # 添加学习率调度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    for epoch in range(num_epochs):
        total_train_loss = 0
        total_recon_loss = 0
        total_kl_loss = 0
        
        for batch_idx, batch in enumerate(train_loader):
            # 处理不同类型的数据批次
            if isinstance(batch, (list, tuple)):
                x = batch[0].to(device)
            else:
                x = batch.to(device)
            
            # 前向传播
            recon_x, mu, logvar, _ = vae(x)
            
            # 重建损失（均方误差）
            recon_loss = F.mse_loss(recon_x, x, reduction='sum') / x.size(0)
            
            # KL散度损失
            kl_loss = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
            
            # 总损失（带权重的KL损失）
            loss = recon_loss + kl_weight * kl_loss
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            # 梯度裁剪防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(vae.parameters(), max_norm=1.0)
            optimizer.step()
            
            # 累计损失
            total_train_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_kl_loss += kl_loss.item()
        
        # 计算平均损失
        avg_train_loss = total_train_loss / len(train_loader)
        avg_recon_loss = total_recon_loss / len(train_loader)
        avg_kl_loss = total_kl_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # 验证
        vae.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                if isinstance(batch, (list, tuple)):
                    x = batch[0].to(device)
                else:
                    x = batch.to(device)
                    
                recon_x, mu, logvar, _ = vae(x)
                
                recon_loss = F.mse_loss(recon_x, x, reduction='sum') / x.size(0)
                kl_loss = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
                loss = recon_loss + kl_weight * kl_loss
                
                total_val_loss += loss.item()
        
        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        # 学习率调度
        scheduler.step(avg_val_loss)
        
        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            # 这里我们将保存路径改为训练函数的外部参数，以便于调用者控制
        
        # 打印详细的训练信息
        print(f'Epoch {epoch+1}/{num_epochs}, '  
              f'Train Loss: {avg_train_loss:.6f}, '  
              f'Recon Loss: {avg_recon_loss:.6f}, '  
              f'KL Loss: {avg_kl_loss:.6f}, '  
              f'Val Loss: {avg_val_loss:.6f}')
        
        vae.train()
    
    return train_losses, val_losses

def create_vae_dataloaders(data, batch_size=64, val_split=0.1):
    """
    为VAE创建数据加载器
    
    Args:
        data: 训练数据
        batch_size: 批次大小
        val_split: 验证集比例
    
    Returns:
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
    """
    import torch.utils.data as data_utils
    
    # 如果数据是numpy数组，转换为tensor
    if isinstance(data, np.ndarray):
        data = torch.tensor(data, dtype=torch.float32)
    
    # 分割训练集和验证集
    val_size = int(len(data) * val_split)
    train_size = len(data) - val_size
    train_data, val_data = data_utils.random_split(data, [train_size, val_size])
    
    # 创建数据加载器
    train_loader = data_utils.DataLoader(
        train_data, 
        batch_size=batch_size, 
        shuffle=True, 
        drop_last=True
    )
    
    val_loader = data_utils.DataLoader(
        val_data, 
        batch_size=batch_size, 
        shuffle=False
    )
    
    return train_loader, val_loader