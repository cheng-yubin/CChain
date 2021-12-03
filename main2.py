import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda

# 训练数据
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

# 测试数据
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

# 神经网络模型
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


def train_loop(dataloader, model, loss_fn, optimizer):
    # 训练数据规模
    size = len(dataloader.dataset)

    # 遍历dataloader中的每一个batch
    for batch, (X, y) in enumerate(dataloader):
        # 计算的预测值与误差
        pred = model(X)
        loss = loss_fn(pred, y)

        # 反向传播，计算梯度
        optimizer.zero_grad()
        loss.backward()

        # 更新模型参数
        optimizer.step()

        # 每100个batch输出一次
        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loos_fn):
    # 测试数据规模
    size = len(dataloader.dataset)

    # batch数量
    num_bataches = len(dataloader)

    # loss, 准确率
    testloss, correct = 0, 0

    # 不需要计算梯度
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            testloss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        testloss /= num_bataches
        correct /= size
        print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss:{testloss:>8f}\n")


if __name__ == "__main__":
    # 超参
    learning_rate = 1e-3    # 学习率
    batch_size = 64         # 每个batch有64条数据
    epochs = 10             # 训练10轮

    # 载入数据
    train_dataloader = DataLoader(training_data, batch_size=64)
    test_dataloader = DataLoader(test_data, batch_size=64)

    # 建立神经网络模型结构
    model = NeuralNetwork()

    # 定义损失函数
    loss_fn = nn.CrossEntropyLoss()

    # 优化器
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # 遍历数据epochs遍
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer)
        test_loop(test_dataloader, model, loss_fn)
    print("Done!")

    torch.save(model, "trained_model.pth")

