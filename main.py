import os
import os.path
import torch 
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.transforms as transforms 
import argparse
import numpy as np
import time

from utils.utils_loss import partial_loss
from utils.models import linear, mlp
from cifar_models import convnet, resnet
from datasets.mnist import mnist
from datasets.fashion import fashion
from datasets.kmnist import kmnist
from datasets.cifar10 import cifar10

torch.manual_seed(0); torch.cuda.manual_seed_all(0) # 设置随机种子

parser = argparse.ArgumentParser(
	prog='PRODEN demo file.',
	usage='Demo with partial labels.',
	description='A simple demo file with MNIST dataset.',
	epilog='end',
	add_help=True)

parser = argparse.ArgumentParser()
parser.add_argument('-lr', help='optimizer\'s learning rate', type=float, default=1e-2)
parser.add_argument('-wd', help='weight decay', type=float, default=1e-3)
parser.add_argument('-bs', help='batch size', type=int, default=256)
parser.add_argument('-ep', help='number of epochs', type=int, default=500)
parser.add_argument('-ds', help='specify a dataset', type=str, default='mnist', choices=['mnist', 'fashion', 'kmnist', 'cifar10'], required=False) # 数据集
parser.add_argument('-model', help='model name', type=str, default='linear', choices=['linear', 'mlp', 'convnet', 'resnet'], required=False) # 模型
parser.add_argument('-decaystep', help='learning rate\'s decay step', type=int, default=10)
parser.add_argument('-decayrate', help='learning rate\'s decay rate', type=float, default=0.9)

parser.add_argument('-partial_type', help='flipping strategy', type=str, default='binomial', choices=['binomial', 'pair']) # 翻转策略
parser.add_argument('-partial_rate', help='flipping probability', type=float, default=0.0, choices=[0.0, 0.1, 0.5, 0.9, 0.7])  # 翻转概率

parser.add_argument('-nw', help='multi-process data loading', type=int, default=4, required=False)
parser.add_argument('-dir', help='result save path', type=str, default='results/', required=False)

parser.add_argument('-name', help='algorithm name', type=str, default='oracle', choices=['PRODEN', 'itera', 'sudden', 'naive', 'oracle'], required=False)

args = parser.parse_args()
 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# load dataset
if args.ds == 'mnist':
    num_features = 28 * 28
    num_classes = 10
    num_training = 60000
    train_dataset = mnist(root='./mnist/',
                                download=True,  
                                train_or_not=True, 
                                transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))]), 
                                partial_type=args.partial_type,
                                partial_rate=args.partial_rate
    )
    test_dataset = mnist(root='./mnist/',
                               download=True,  
                               train_or_not=False, 
                               transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))]),
                               partial_type=args.partial_type,
                               partial_rate=args.partial_rate
    )

if args.ds == 'fashion':
    num_features = 28 * 28
    num_classes = 10
    num_training = 60000
    train_dataset = fashion(root='./fashion/',
                                download=True,  
                                train_or_not=True, 
                                transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))]), 
                                partial_type=args.partial_type,
                                partial_rate=args.partial_rate
    ) 
    test_dataset = fashion(root='./fashion/',
                               download=True,  
                               train_or_not=False, 
                               transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))]),
                               partial_type=args.partial_type,
                               partial_rate=args.partial_rate
    )

if args.ds == 'kmnist':
    num_features = 28 * 28
    num_classes = 10
    num_training = 60000
    train_dataset = kmnist(root='./kmnist/',
                                download=True,  
                                train_or_not=True, 
                                transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,), (0.5,))]), 
                                partial_type=args.partial_type,
                                partial_rate=args.partial_rate
    ) 
    test_dataset = kmnist(root='./kmnist/',             
                               download=True,  
                               train_or_not=False, 
                               transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,), (0.5,))]), 
                               partial_type=args.partial_type,
                               partial_rate=args.partial_rate
    )

if args.ds == 'cifar10':
    input_channels = 3
    num_classes = 10
    dropout_rate = 0.25
    num_training = 50000
    train_dataset = cifar10(root='./cifar10/',
                                download=True,  
                                train_or_not=True, 
                                transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),]),
                                partial_type=args.partial_type,
                                partial_rate=args.partial_rate
    ) 
    test_dataset = cifar10(root='./cifar10/',
                                download=True,  
                                train_or_not=False, 
                                transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),]),
                                partial_type=args.partial_type,
                                partial_rate=args.partial_rate
    )


# learning rate 
lr_plan = [args.lr] * args.ep 
for i in range(0, args.ep):
    lr_plan[i] = args.lr * args.decayrate ** (i // args.decaystep)

def adjust_learning_rate(optimizer, epoch):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_plan[epoch]


# result dir  
save_dir = './' + args.dir
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
save_file = os.path.join(save_dir, (args.name + '_' + args.ds + '_' + args.model + '_' + args.partial_type + '_' + str(args.partial_rate) + '.txt'))

# calculate accuracy
def evaluate(loader, model):
    model.eval()     
    correct = 0
    total = 0
    for images, _, labels, _ in loader:
        images = images.to(device)
        labels = labels.to(device)
        output1 = model(images)
        output = F.softmax(output1, dim=1)
        _, pred = torch.max(output.data, 1) 
        total += images.size(0)
        correct += (pred == labels).sum().item()
    acc = 100 * float(correct) / float(total)
    return acc


def main():
    # print ('loading dataset...')
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=args.bs, 
                                               num_workers=args.nw,
                                               drop_last=True, 
                                               shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=args.bs, 
                                              num_workers=args.nw,
                                              drop_last=False,
                                              shuffle=False)

    # print ('building model...')
    if args.model == 'linear':
        net = linear(n_inputs=num_features, n_outputs=num_classes)
    elif args.model == 'mlp':
        net = mlp(n_inputs=num_features, n_outputs=num_classes)
    elif args.model == 'convnet':
        net = convnet(input_channels=input_channels, n_outputs=num_classes, dropout_rate=dropout_rate)
    elif args.model == 'resnet':
        net = resnet(depth=32, n_outputs=num_classes)
    net.to(device)
    print (net.parameters)

    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, weight_decay=args.wd, momentum=0.9)

    for epoch in range(0, args.ep):
        # print ('training...')
        net.train()   
        adjust_learning_rate(optimizer, epoch)

        for i, (images, labels, trues, indexes) in enumerate(train_loader):   # 一个train_loader 每次循环输出batchsize = 256数据
            images = Variable(images).to(device)  # 
            labels = Variable(labels).to(device)
            trues = trues.to(device)
            output = net(images)   # 输出10类 torch,Size([256,10])
            
            loss = partial_loss(output, labels, trues)   # new_label 大小 torch.Size([256, 10])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            # for j, k in enumerate(indexes):
            #     train_loader.dataset.train_final_labels[k,:] = new_label[j,:].detach()  # detach 简单的数据复制，既不数据共享，也不对梯度共享
            # PRODEN-itera 每100epoch 更新weights
            # if (epoch + 1) % 100 == 0:
            #     for j, k in enumerate(indexes):
            #         train_loader.dataset.train_final_labels[k,:] = new_label[j,:].detach()  # detach 简单的数据复制，既不数据共享，也不对梯度共享
 
        # print ('evaluating model...')       
        train_acc = evaluate(train_loader, net)
        test_acc = evaluate(test_loader, net)

        print(str(int(epoch)) + ': Training Acc.: ' + str(round(train_acc, 4)) + ' , Test Acc.: ' + str(round(test_acc, 4)) + '\n')

        with open(save_file, 'a') as file:
            file.write(str(int(epoch)) + ': Training Acc.: ' + str(round(train_acc, 4)) + ' , Test Acc.: ' + str(round(test_acc, 4)) + '\n')
    

if __name__=='__main__':
    main()
 