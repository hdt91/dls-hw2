import sys
sys.path.append('../python')
import needle as ndl
import needle.nn as nn
import numpy as np
import time
import os

np.random.seed(0)

def ResidualBlock(dim, hidden_dim, norm=nn.BatchNorm1d, drop_prob=0.1):
    ### BEGIN YOUR SOLUTION
    return nn.Sequential(
        nn.Residual(nn.Sequential(
            nn.Linear(dim, hidden_dim),
            norm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(drop_prob),
            nn.Linear(hidden_dim, dim),
            norm(dim),
        )),
        nn.ReLU())
    ### END YOUR SOLUTION


def MLPResNet(dim, hidden_dim=100, num_blocks=3, num_classes=10, norm=nn.BatchNorm1d, drop_prob=0.1):
    ### BEGIN YOUR SOLUTION
    blocks = [
        nn.Linear(dim, hidden_dim),
        nn.ReLU(),
    ]

    for _ in range(num_blocks):
        blocks.append(ResidualBlock(hidden_dim, hidden_dim//2, norm=norm, drop_prob=drop_prob))

    blocks.append(nn.Linear(hidden_dim, num_classes))

    return nn.Sequential(*blocks)
    ### END YOUR SOLUTION




def epoch(dataloader, model, opt=None):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    if opt:
        model.train()
    else:
        model.eval()

    counter = 0
    total_loss = 0
    total_error = 0
    img_size = 28*28
    for i, batch in enumerate(dataloader):
        counter += 1

        if opt:
            opt.reset_grad()

        x_batch = batch[0].numpy()
        batch_size = x_batch.size // img_size
        # x_batch = x_batch.reshape(img_size, batch_size)
        # x_batch = ndl.Tensor(np.transpose(x_batch))
        x_batch = ndl.Tensor(x_batch.reshape(batch_size, img_size))
        y_batch = ndl.Tensor(batch[1].numpy())
        h = model(x_batch)
        loss = nn.SoftmaxLoss()(h, y_batch)
        if opt:
            loss.backward()
            opt.step()

        total_loss += loss.numpy()
        error = np.mean(h.numpy().argmax(axis=1))
        total_error += error

    a = [total_error/counter, total_loss/counter]
    print("RET", a)
    return a

    ### END YOUR SOLUTION



def train_mnist(batch_size=100, epochs=10, optimizer=ndl.optim.Adam,
                lr=0.001, weight_decay=0.001, hidden_dim=100, data_dir="data"):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    train_dataset = ndl.data.MNISTDataset(\
            f"./{data_dir}/train-images-idx3-ubyte.gz",
            f"./{data_dir}/train-labels-idx1-ubyte.gz")
    train_dataloader = ndl.data.DataLoader(\
             dataset=train_dataset,
             batch_size=batch_size,
             shuffle=True)

    model = MLPResNet(784, hidden_dim)
    opt = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)

    for _ in range(epochs):
        train_error, train_loss = epoch(train_dataloader, model, opt=opt)

    test_dataset = ndl.data.MNISTDataset(\
            f"./{data_dir}/t10k-images-idx3-ubyte.gz",
            f"./{data_dir}/t10k-labels-idx1-ubyte.gz")
    test_dataloader = ndl.data.DataLoader(\
             dataset=test_dataset,
             batch_size=batch_size,
             shuffle=True)
    test_error, test_loss = epoch(test_dataloader, model, opt=None)

    return train_error, train_loss, test_error, test_loss
    ### END YOUR SOLUTION



if __name__ == "__main__":
    train_mnist(data_dir="../data")
