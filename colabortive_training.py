import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda
from web3 import Web3
import time
import numpy as np


def gradient_upload(w3, average_contract, it_now, grad_int_list):
    w3.geth.personal.unlock_account(w3.eth.default_account, '88039983cyb')
    print(f"iteration: {it_now}")
    print(f"{time.time()}  upload...")
    tx_hash = average_contract.functions.upload_gradient(it_now, grad_int_list).transact()
    w3.eth.wait_for_transaction_receipt(tx_hash)
    print("upload finished")


def gradient_download(w3, average_contract, transform_scale):
    w3.geth.personal.unlock_account(w3.eth.default_account, '88039983cyb')
    print("download...")
    gradient_int = average_contract.functions.get_average_view().call()

    tx_hash = average_contract.functions.get_average().transact()
    w3.eth.wait_for_transaction_receipt(tx_hash)
    # print(f"downloaded gradient (int): {gradient_int}")
    print("download finished")
    return gradient_int


def get_average_from_chain(grad_int_list, w3, average_contract, gradient_size, transform_scale, steps):
    print("start to calculate the average")
    average_gradient_int = []
    # 按照批次发送
    for i in range(steps):
        print(f"{i} / {steps}")
        # if idx == '0':
        #     time.sleep(5)
        upload_list = grad_int_list[i * gradient_size: (i + 1) * gradient_size]
        while True:
            state = average_contract.functions.get_contract_state().call()
            # 上传状态
            if not state:
                has_upload = average_contract.functions.get_has_upload().call()
                if has_upload:
                    time.sleep(0.1)
                    continue
                else:
                    it_now = average_contract.functions.get_iteration().call()
                    gradient_upload(w3, average_contract, it_now, upload_list)
            else:
                has_download = average_contract.functions.get_has_download().call()
                if has_download:
                    time.sleep(0.1)
                    continue
                else:
                    download_list = gradient_download(w3, average_contract, transform_scale)
                    average_gradient_int = average_gradient_int + download_list
                    print("\n\n")
                    break
    return average_gradient_int


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


def train_loop(dataloader, model, loss_fn, optimizer, w3, average_contract, transform_scale, gradient_size):
    # 训练数据规模
    size = len(dataloader.dataset)

    # 遍历dataloader中的每一个batch
    for batch, (X, y) in enumerate(dataloader):
        # 计算的预测值与误差
        pred = model(X)
        loss = loss_fn(pred, y)
        model.named_parameters()
        # 反向传播，计算梯度
        optimizer.zero_grad()
        loss.backward()

        # 获得梯度，展平，转为int列表
        print(f"batch: {batch}")
        flattened_grad = torch.tensor([])
        total_size = 0
        for _, param in enumerate(model.parameters()):
            # print(f"original grad: {param.grad}")
            total_size += param.shape.numel()
            flattened_grad = torch.cat([flattened_grad, torch.flatten(param.grad)])
        grad_int_list = (flattened_grad * transform_scale).numpy().astype(int).tolist()

        # 计算上传批次数量
        steps = 0
        n_0 = 0
        if total_size % gradient_size == 0:
            steps = total_size // gradient_size
            n_0 = 0
        else:
            steps = total_size // gradient_size + 1
            n_0 = gradient_size * steps - total_size
        # 将grad_int_list补齐为 gradient_size 整数倍
        grad_int_list = grad_int_list + [0] * n_0

        # 从链上获得平均值
        aver_grad_int_list = get_average_from_chain(grad_int_list, w3, average_contract, gradient_size, transform_scale,
                                                    steps)

        aver_grad = torch.tensor(aver_grad_int_list, dtype=int)
        aver_grad = aver_grad * 1.0 / transform_scale

        for _, param in enumerate(model.parameters()):
            shape = param.grad.shape
            num = shape.numel()
            grad_flatten, aver_grad = aver_grad.split([num, aver_grad.numel() - num], dim=0)
            grad_ = grad_flatten.reshape(shape)
            param.grad = grad_

        for _, param in enumerate(model.parameters()):
            print(f"new grad: {param.grad}")

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
    idx = input("input party idx: ")

    # 智能合约准备 ----------------------------------------------------------------------------------------
    http_port = ''
    if idx == '0':
        http_port = 'http://127.0.0.1:8545'
        id_ = 0  # 数据集分割标志
    elif idx == '1':
        id_ = 1
        http_port = 'http://127.0.0.1:8101'
    print(f"http_port: {http_port}")

    # 连接 provider
    w3 = Web3(Web3.HTTPProvider(http_port))  # 连接local provider
    if w3.isConnected():
        print("geth connected")
    else:
        print("fail to connect")
        exit()

    # 设置默认账户并解锁
    w3.eth.default_account = w3.eth.accounts[0]
    w3.geth.personal.unlock_account(w3.eth.default_account, '88039983cyb')
    print(f"account: {w3.eth.default_account}")

    # address & abi
    address_contract = "0xfA62cE60420744A0bdD4c47718d50D00a361bfC3"
    abi = [{'inputs': [{'internalType': 'uint256',
                        'name': 'gradient_size',
                        'type': 'uint256'},
                       {'internalType': 'uint256', 'name': 'scale', 'type': 'uint256'},
                       {'internalType': 'address[]',
                        'name': 'parties_address',
                        'type': 'address[]'}],
            'stateMutability': 'nonpayable',
            'type': 'constructor'},
           {'inputs': [{'internalType': 'uint256', 'name': '', 'type': 'uint256'}],
            'name': 'average',
            'outputs': [{'internalType': 'int256', 'name': '', 'type': 'int256'}],
            'stateMutability': 'view',
            'type': 'function'},
           {'inputs': [],
            'name': 'ctrl_params',
            'outputs': [{'internalType': 'uint256',
                         'name': 'iteration',
                         'type': 'uint256'},
                        {'internalType': 'uint256', 'name': 'gradient_size', 'type': 'uint256'},
                        {'internalType': 'uint256', 'name': 'scale', 'type': 'uint256'},
                        {'internalType': 'bool', 'name': 'upload_or_download', 'type': 'bool'}],
            'stateMutability': 'view',
            'type': 'function'},
           {'inputs': [],
            'name': 'get_average',
            'outputs': [],
            'stateMutability': 'nonpayable',
            'type': 'function'},
           {'inputs': [],
            'name': 'get_average_view',
            'outputs': [{'internalType': 'int256[]', 'name': '', 'type': 'int256[]'}],
            'stateMutability': 'view',
            'type': 'function'},
           {'inputs': [],
            'name': 'get_contract_state',
            'outputs': [{'internalType': 'bool', 'name': '', 'type': 'bool'}],
            'stateMutability': 'view',
            'type': 'function'},
           {'inputs': [],
            'name': 'get_gradient_size',
            'outputs': [{'internalType': 'uint256', 'name': '', 'type': 'uint256'}],
            'stateMutability': 'view',
            'type': 'function'},
           {'inputs': [],
            'name': 'get_has_download',
            'outputs': [{'internalType': 'bool', 'name': '', 'type': 'bool'}],
            'stateMutability': 'view',
            'type': 'function'},
           {'inputs': [],
            'name': 'get_has_permitted',
            'outputs': [{'internalType': 'bool', 'name': '', 'type': 'bool'}],
            'stateMutability': 'view',
            'type': 'function'},
           {'inputs': [],
            'name': 'get_has_upload',
            'outputs': [{'internalType': 'bool', 'name': '', 'type': 'bool'}],
            'stateMutability': 'view',
            'type': 'function'},
           {'inputs': [],
            'name': 'get_iteration',
            'outputs': [{'internalType': 'uint256', 'name': '', 'type': 'uint256'}],
            'stateMutability': 'view',
            'type': 'function'},
           {'inputs': [],
            'name': 'get_transform_scale',
            'outputs': [{'internalType': 'uint256', 'name': '', 'type': 'uint256'}],
            'stateMutability': 'view',
            'type': 'function'},
           {'inputs': [{'internalType': 'address', 'name': '', 'type': 'address'}],
            'name': 'parties',
            'outputs': [{'internalType': 'bool',
                         'name': 'isPermissible',
                         'type': 'bool'},
                        {'internalType': 'bool', 'name': 'has_uploaded', 'type': 'bool'},
                        {'internalType': 'bool', 'name': 'has_download', 'type': 'bool'}],
            'stateMutability': 'view',
            'type': 'function'},
           {'inputs': [],
            'name': 'reset',
            'outputs': [],
            'stateMutability': 'nonpayable',
            'type': 'function'},
           {'inputs': [{'internalType': 'uint256', 'name': 'it', 'type': 'uint256'},
                       {'internalType': 'int256[]', 'name': 'num', 'type': 'int256[]'}],
            'name': 'upload_gradient',
            'outputs': [],
            'stateMutability': 'nonpayable',
            'type': 'function'}]

    print(f"address: {address_contract}")
    print(f"address: {abi}")

    # 创建合约对象
    average_contract = w3.eth.contract(address=address_contract, abi=abi)

    # 检查账户授权
    permitted = average_contract.functions.get_has_permitted().call()
    if permitted:
        print("This address is permitted")
    else:
        print("No permission")
        exit()

    # float 转 int 缩放比例
    transform_scale = average_contract.functions.get_transform_scale().call()
    print(f"transform_scale: {transform_scale}")

    # gradient_size 单次上传的大小
    gradient_size = average_contract.functions.get_gradient_size().call()
    print(f"gradient_size: {gradient_size}")

    # 神经网络准备工作  ---------------------------------------------------------------------------------------------------
    # 训练数据
    training_data = datasets.FashionMNIST(root="data", train=True, download=True, transform=ToTensor())
    id_list = list(range(id_, len(training_data), 2))
    training_subset = torch.utils.data.Subset(training_data, id_list)
    # 测试数据
    test_data = datasets.FashionMNIST(root="data", train=False, download=True, transform=ToTensor())

    # 超参
    learning_rate = 1e-3  # 学习率
    batch_size = 64  # 每个batch有64条数据
    epochs = 10  # 训练10轮

    # 载入数据
    train_dataloader = DataLoader(training_subset, batch_size=64)
    test_dataloader = DataLoader(test_data, batch_size=64)

    # 建立神经网络模型结构
    model = NeuralNetwork()

    # 定义损失函数
    loss_fn = nn.CrossEntropyLoss()

    # 优化器
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # 遍历数据epochs遍
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------")
        # 协作训练
        train_loop(train_dataloader, model, loss_fn, optimizer, w3, average_contract, transform_scale, gradient_size)
        # 单独测试
        test_loop(test_dataloader, model, loss_fn)
    print("Done!")

    torch.save(model, "trained_model.pth")
