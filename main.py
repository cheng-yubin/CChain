import torch
from torch import nn
from torch.nn import functional
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor, Lambda
from torchvision.io import read_image
import matplotlib.pyplot as plt
import os
import pandas as pd

training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
    target_transform=Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))

)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
    target_transform=Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))
)

labels_map = {
    0: "T-shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}


def show_data_samples():
    figure = plt.figure(figsize=(8, 8))
    cols, rows = 3, 3
    for i in range(1, cols * rows + 1):
        sample_idx = torch.randint(len(training_data), size=(1,)).item()
        img, label = training_data[sample_idx]
        figure.add_subplot(rows, cols, i)
        plt.title(labels_map[label])
        plt.axis("off")
        plt.imshow(img.squeeze(), cmap="gray")
    plt.show()


def display_image_label(data_loader):
    # display image and label.
    train_features, train_labels = next(iter(data_loader))
    print(f"Features batch size: {train_features.size()}")
    print(f"Labels batch size: {train_labels.size()}")

    img = train_features[0].squeeze()
    label = train_labels[0]
    plt.imshow(img, cmap="gray")
    plt.show()
    print(f"Label: {label}")


class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


if __name__ == "__main__":
    # show_data_samples()

    # choose device
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # print(f"Using {device} device")

    # data loader
    # train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
    # test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

    # display_image_label(train_dataloader)

    # network model
    # model = NeuralNetwork().to(device)
    # print(model)

    # forward pass test
    # X = torch.rand((1, 28, 28), device=device)
    # logits = model(X)
    # print(logits)
    # pred_probab = nn.Softmax(dim=1)(logits)
    # print(pred_probab)
    # y_pred = pred_probab.argmax(1)
    # print(f"Predicted class: {y_pred}")

    # model layers
    # input_image = torch.rand((3, 28, 28), device=device)
    # print(input_image.size())
    #
    # flatten = nn.Flatten()
    # flat_image = flatten(input_image)
    # print(flat_image.size())
    #
    # layer1 = nn.Linear(in_features=28*28, out_features=20)
    # hidden1 = layer1(flat_image)
    # print(hidden1.size())
    #
    # print(f"Before Relu: {hidden1}")
    # hidden1 = nn.ReLU()(hidden1)
    # print(f"After Relu: {hidden1}")
    #
    # seq_modules = nn.Sequential(
    #     flatten,
    #     layer1,
    #     nn.ReLU(),
    #     nn.Linear(20, 10)
    # )
    # logits = seq_modules(input_image)
    # softmax = nn.Softmax(dim=1)
    # pred_probab = softmax(logits)
    #
    # print(f"pred_probab: {pred_probab}")
    #
    # print("model structure ", model, "\n\n")
    # for name, param in model.named_parameters():
    #     print(f"Layer: {name} | size: {param.size()} | Values: {param[:2]} \n")

    x = torch.ones(5)
    y = torch.zeros(3)
    w = torch.randn((5, 3), requires_grad=True)
    b = torch.randn(3, requires_grad=True)
    z = torch.matmul(x, w) + b
    loss = functional.binary_cross_entropy_with_logits(z, y)

    print("Gradient function for z=", z.grad_fn)
    print("Gradient function for loss =", loss.grad_fn)

    loss.backward()
    print(w.grad)
    print(b.grad)

    w_grad = torch.rand_like(w.grad)
    b_grad = torch.rand_like(b.grad)
    print(w_grad)
    print(b_grad)
    w.grad = w_grad
    b.grad = b_grad
    print(w.grad)
    print(b.grad)

    w_grad_list = w.grad.numpy().tolist()
    b_grad_list = b.grad.numpy().tolist()
    print("w_grad_list", w_grad_list)
    print("b_grad_list", b_grad_list)

    print(z.requires_grad)
    z_det = z.detach()
    print(z_det.requires_grad)
