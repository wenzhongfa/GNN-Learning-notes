import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_networkx
import networkx as nx
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


#####################################
# 数据准备
#####################################
# 通过Planetoid下载数据
dataset = Planetoid(
    # 数据保存的路径
    # 如果指定目录下不存在数据集
    # PyG会自己进行下载
    root='../data/Cora',
    # 要使用的数据集
    name='Cora',
)
# 查看数据集基本信息
print(dataset.data)
print('包含的类别数:', dataset.num_classes)
print('边特征的维度:', dataset.num_edge_features)
print('边的数量:', dataset.data.edge_index.shape[1] / 2)
print('节点特征的维度：', dataset.num_node_features)
print('节点的数量:', dataset.data.x.shape[0])
print('节点属性特征:', dataset.data.x)

# 查看网络数据中节点的连接方式
# 列表中的每一对代表对应节点之间有边相连接，比如0-633，0-1862
print(dataset.data.edge_index)

# 获取训练集、测试集、验证集数据量
print('训练集节点数量:', sum(dataset.data.train_mask))
print('验证集节点数量:', sum(dataset.data.val_mask))
print('测试集节点数量:', sum(dataset.data.test_mask))

# 检查数据集是否是无向图
print(dataset.data.is_undirected())

# 将数据集转换为网络的形式方便可视化处理
CoraNet = to_networkx(dataset.data)
# 转换为无向图，因为前面我们判断出来Cora是无向图
CoraNet = CoraNet.to_undirected()
# 在网络形式上获取数据集的信息
print('网络中边的数量:', CoraNet.number_of_edges())
print('网络中节点的数量:', CoraNet.number_of_nodes())
# 每个节点的类别
Node_class = dataset.data.y.data.numpy()
print(Node_class)
# 获取每个节点的度
Node_degree = pd.DataFrame(
    data=CoraNet.degree,
    columns=['Node', 'Degree']
)
# 降序排列
Node_degree = Node_degree.sort_values(
    by=['Degree'],
    # 降序
    ascending=False
)
# 因为排序后索引会改变，重置一下索引
Node_degree = Node_degree.reset_index(drop=True)
# 使用直方图可视化度最多的前30个节点
# 代表了被引用最多
# 说明节点上代表的论文是比较重要的
Node_degree.iloc[0:30, :].plot(x='Node', y='Degree', kind='bar', figsize=(10, 10))
plt.xlabel('Node', size=12)
plt.ylabel('Degree', size=12)
plt.show()
# 可视化训练集数据
# 网络中每个节点的位置
pos = nx.spring_layout(CoraNet)
# 为不同类别的节点赋予不同的颜色
nodecolor = [
    'red',
    'blue',
    'green',
    'yellow',
    'peru',
    'violet',
    'cyan',
]
# 训练数据集数据的索引
nodelabel = np.arange(0, 140)
# 对应的类别
Node_class = dataset.data.y.data.numpy()[0:140]
plt.figure(figsize=(10, 10))
# np.unique(Node_class)表示获取训练集的类别
# len()表示获取类别数
# np.arange()表示生成从0到类别数的索引
for ii in np.arange(len(np.unique(Node_class))):
    # 获取相应类别的节点
    nodelist = nodelabel[Node_class == ii]
    # 绘图
    nx.draw_networkx_nodes(
        # 要画的图
        CoraNet,
        # 每个节点在图中的位置
        pos,
        # 具体要画哪些节点
        nodelist=list(nodelist),
        # 节点大小
        node_size=50,
        # 节点颜色
        node_color=nodecolor[ii],
        # 透明度
        alpha=0.8,
    )
plt.show()


##############################################
# 搭建图卷积神经网络
##############################################
class GCNNet(torch.nn.Module):
    # 网络初始化
    def __init__(self, input_feature, num_class):
        '''
        :param input_feature: 每个节点属性的维度
        :param num_class: 节点所属的类别数
        '''
        super(GCNNet, self).__init__()
        self.input_feature = input_feature
        self.num_classes = num_class
        # 图卷积层
        # 出自于论文《Semi-Supervised Classification with Graph Convolutional Networks》
        # 具体计算公式可查阅论文
        self.conv1 = GCNConv(input_feature, 32)
        self.conv2 = GCNConv(32, num_class)

    # 前向传递过程
    def forward(self, data):
        # data是输入到网络中的训练数据
        # 获取节点的属性信息和边的连接信息
        x, edge_index = data.x, data.edge_index
        # 卷积，利用相邻节点的信息更新自己的特征表示
        x = self.conv1(x, edge_index)
        # 激活
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        # 预测类别
        output = F.softmax(x, dim=1)
        return output


# 初始化网络
input_feature = dataset.num_node_features
num_class = dataset.num_classes
mygcn = GCNNet(input_feature, num_class)

####################################################################
# 训练图卷积神经网络
####################################################################
# 当前设备
# 如果GPU可用用GPU，否则用CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 将模型和训练数据放在设备上
model = mygcn.to(device)
data = dataset[0].to(device)
# Adam优化器
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
# 损失值
train_loss_all = []
val_loss_all = []
# 开启训练
model.train()
print('图卷积神经网络开始训练！')
# 训练200轮
for epoch in range(200):
    print('=' * 100)
    print('当前训练轮次:', epoch)
    # 梯度清零
    optimizer.zero_grad()
    # 获取模型对于数据的类别预测
    out = model(data)
    # 计算损失
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    # 反向传播
    loss.backward()
    optimizer.step()
    train_loss_all.append(loss.data.numpy())

    # 计算在验证集上的损失
    loss = F.cross_entropy(out[data.val_mask], data.y[data.val_mask])
    val_loss_all.append(loss.data.numpy())
    print('Epoch:{},train_loss:{},val_loss:{}'.format(epoch, train_loss_all[-1], val_loss_all[-1]))


# 保存训练完毕的网络
torch.save(model, 'GCNNet.pt')