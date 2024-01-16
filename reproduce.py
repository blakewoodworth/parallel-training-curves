import numpy as np
import os
import random
import torch
import torchvision
from tqdm import tqdm
import torchvision.transforms as transforms
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid", context="talk", palette=sns.color_palette("bright"), color_codes=False)
matplotlib.rcParams['mathtext.fontset'] = 'cm'
np.set_printoptions(precision=3, linewidth=160)

####################################################################################################

def seed_everything(seed=2024):
    '''
    Set the seed for reproducibility.
    '''
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def load_data(dataset='cifar10', batch_size=128, data_order_seed=1993, num_workers=2):
    """
    Loads the required dataset
    :param dataset: Can be either 'cifar10' or 'cifar100'
    :param batch_size: The desired batch size
    :return: Tuple (train_loader, test_loader, num_classes)
    """
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    if dataset == 'cifar10':
        # classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        num_classes = 10
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    elif dataset == 'cifar100':
        num_classes = 100
        trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    else:
        raise ValueError('Only cifar 10 and cifar 100 are supported')

    torch.manual_seed(data_order_seed)
    generator = torch.Generator()
    generator.manual_seed(1)
    data_rng = torch.utils.data.RandomSampler(data_source=trainset, generator=generator)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, sampler=data_rng, num_workers=num_workers, pin_memory=True)

    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return trainloader, testloader, num_classes

def accuracy_and_loss(net, dataloader, device, criterion):
    net.eval()
    correct = 0
    total = 0
    loss = 0
    with torch.no_grad():
        for data in tqdm(dataloader, leave=False):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            loss += criterion(outputs, labels).cpu().item() / len(dataloader)

    return 100 * correct / total, loss


####################################################################################################

model_init_seed = 2024
data_seed = 1993
dataset = 'cifar10'
model_type = 'resnet18'
batch_size = 128
num_workers = 0

learning_rate = 0.01
momentum = 0.9
n_epoch = 2

####################################################################################################

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Device: {}'.format(device))

seed_everything(model_init_seed)
net = torch.hub.load('pytorch/vision:v0.10.0', model_type, pretrained=False)
net.to(device)
net.train()
criterion = torch.nn.CrossEntropyLoss()

trainloader, testloader, num_classes = load_data(dataset=dataset, batch_size=batch_size, data_order_seed=data_seed, num_workers=num_workers)
N_train = len(trainloader)*batch_size
checkpoint = len(trainloader) // 3 + 1

optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epoch)

total = 0
for w in net.parameters():
    total += np.prod(w.shape)
print('Total number of parameters: {}'.format(total))

test_losses = []
test_acc = []
it_test = []

for epoch in tqdm(range(n_epoch)):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in tqdm(enumerate(trainloader), total=len(trainloader), leave=False):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad(set_to_none=True)

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if i % checkpoint == checkpoint - 1:
            running_loss = 0.0
            test_a, test_l = accuracy_and_loss(net, testloader, device, criterion)
            test_acc.append(test_a)
            test_losses.append(test_l)
            net.train()
            it_test.append(epoch + i * batch_size / N_train)
    scheduler.step()

fig, ax = plt.subplots(figsize=(7,7), tight_layout=True)
ax.plot(it_test, test_losses, color='green')
plt.savefig('loss_curve.pdf')








