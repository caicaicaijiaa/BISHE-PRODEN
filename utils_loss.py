import torch 
import torch.nn.functional as F
import numpy as np

def partial_loss(output1, target, true):
    """
    这里传进来的数据是一个batchsize
    对应main.py中的 output, labels, trues
    output1: net的输出 MNIST ： torch.Size([256,10]) 预测值 
    target：翻转后的标记 torch.Size([256,10])
    true: 真实标记，里面只有一个位置是1 torch.Size([256])
    """
    output = F.softmax(output1, dim=1)   # torch.Size([256,10])
    l = target * torch.log(output)
    loss = (-torch.sum(l)) / l.size(0)  # 交叉熵损失

    # revisedY = target.clone()

    # revisedY[revisedY > 0]  = 1  # 权重值是小数（10） 变成1
    # revisedY = revisedY * output  # 计算新的权重 用新的模型输出与对应的类别标记 使用当前的预测对更多可能的标签给更多权重
    
    # PRODEN

    # revisedY = revisedY / revisedY.sum(dim=1).repeat(revisedY.size(1),1).transpose(0,1)  
    
    """
    256,10 -> 256 -> 10,256 -> 256,10
    压缩dim =1 得到256个数 是每个样本的分数和 -> 10,256 10行都是一样的
    转置 使得每行10个数都是对应样本的分数和 都一样 一共256 行
    最终 gj(xi)/每行的和 每个标记的值除以总和
    """

    # PRODEN-sudden
    n, c = revisedY.shape[0],revisedY.shape[1]
    a = torch.zeros(c).cuda()
    b = torch.ones(c).cuda() 

    for i in range(n):
        row = revisedY[i,:]
        revisedY[i,:] = torch.where(row < row.max(), a, b)
        
    new_target = revisedY  # torch.Size([256,10])


    return loss
