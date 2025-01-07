import torch
import torch.nn as nn

"""
对比学习loss
v1: 对比音频和文本的embedding
v2: 对比音频和文本的embedding的均值
"""

class ContrastiveLossV1(nn.Module):
    def __init__(self, margin=0.2, max_violation=False, proj_dim=128):
        super(ContrastiveLossV1, self).__init__()
        self.margin = margin
        self.max_violation = max_violation
        self.audio_proj = nn.Linear(2048, proj_dim)       # 将音频特征降维到 `proj_dim`
        self.text_proj = nn.Linear(128257, proj_dim)    # 将文本特征降维到 `proj_dim`

    def forward(self, audio_embeddings, text_embeddings):
        # 降维，将音频和文本投影到同一特征空间
        audio_embeddings = self.audio_proj(audio_embeddings)  # [batch, length, proj_dim]
        text_embeddings = self.text_proj(text_embeddings)     # [batch, length, proj_dim]
        
        # 计算相似度矩阵，先 reshape 到 [batch * length, proj_dim]
        audio_flat = audio_embeddings.view(-1, audio_embeddings.size(-1))
        text_flat = text_embeddings.view(-1, text_embeddings.size(-1))
        
        # 计算嵌入相似度矩阵 [batch * length, batch * length]
        scores = torch.matmul(audio_flat, text_flat.t())
        
        # 对角线上的元素（正样本）代表同一个位置的音频和文本
        diagonal = scores.diag().view(audio_flat.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        # 对比正样本与负样本
        cost_audio_to_text = (self.margin + scores - d1).clamp(min=0)
        cost_text_to_audio = (self.margin + scores - d2).clamp(min=0)
        
        # 清除对角线元素
        mask = torch.eye(scores.size(0), dtype=torch.bool, device=scores.device)
        cost_audio_to_text = cost_audio_to_text.masked_fill_(mask, 0)
        cost_text_to_audio = cost_text_to_audio.masked_fill_(mask, 0)

        # 最大违背策略
        if self.max_violation:
            cost_audio_to_text, _ = cost_audio_to_text.max(1)
            cost_text_to_audio, _ = cost_text_to_audio.max(0)

        # 计算总损失
        loss = cost_audio_to_text.sum() + cost_text_to_audio.sum()
        return loss


class ContrastiveLossV2(nn.Module):
    def __init__(self, margin=0.2, max_violation=False):
        super(ContrastiveLossV2, self).__init__()
        self.margin = margin
        self.max_violation = max_violation

    def forward(self, audio_embeddings, text_embeddings):
        # 在最后一维上取均值，得到 [batch, length]
        audio_embeddings = audio_embeddings.mean(dim=-1)
        text_embeddings = text_embeddings.mean(dim=-1)

        # 计算相似度矩阵
        scores = torch.matmul(audio_embeddings, text_embeddings.t())

        # 对角线元素代表正样本
        diagonal = scores.diag().view(audio_embeddings.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        # 对比正样本和负样本
        cost_audio_to_text = (self.margin + scores - d1).clamp(min=0)
        cost_text_to_audio = (self.margin + scores - d2).clamp(min=0)

        # 清除对角线元素
        mask = torch.eye(scores.size(0), dtype=torch.bool, device=scores.device)
        cost_audio_to_text = cost_audio_to_text.masked_fill_(mask, 0)
        cost_text_to_audio = cost_text_to_audio.masked_fill_(mask, 0)

        # 最大违背策略
        if self.max_violation:
            cost_audio_to_text, _ = cost_audio_to_text.max(1)
            cost_text_to_audio, _ = cost_text_to_audio.max(0)

        # 计算总损失
        loss = cost_audio_to_text.sum() + cost_text_to_audio.sum()
        return loss



if __name__ == '__main__':
    loss = ContrastiveLoss()
    audio_embeddings = torch.randn(4, 20)
    text_embeddings = torch.randn(4, 20)
    loss = loss(audio_embeddings, text_embeddings)
    print(loss)