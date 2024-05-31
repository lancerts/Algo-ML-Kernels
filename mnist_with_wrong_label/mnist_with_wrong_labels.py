#Context: https://www.youtube.com/watch?v=n4IQOBka8bc&ab_channel=Sana (15 min mark)
#Experiment: Verify the test accuracy when X% of the training data label is wrong consistently during different epochs 
#Outcome: when 45% of the training label is WRONG, the test error is ~8% in 15 epoch.
#Aligned with the claim in the video, model can learn better than the teachers.

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR


class IndexedDataset(torch.utils.data.Dataset):
    def __init__(self, original_dataset):
        self.original_dataset = original_dataset

    def __getitem__(self, index):
        data = self.original_dataset[index]
        return index, data

    def __len__(self):
        return len(self.original_dataset)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


import random

random.seed(123)


def generate_wrong_label(input_number):
    if not 0 <= input_number <= 9:
        raise ValueError("Input number must be between 0 and 9")

    numbers = list(range(0, 10))
    numbers.remove(input_number)
    return random.choice(numbers)


# def generate_wrong_labels_map(input_labels):
#     wrong_labels_map = {input_label: generate_wrong_label(input_label) for input_label in input_labels}
#     return wrong_labels_map


def generate_wrong_labels_map(input_labels):
    shuffled_labels = input_labels.copy()
    random.shuffle(shuffled_labels)
    random_permutation_map = {original: shuffled for original, shuffled in zip(input_labels, shuffled_labels)}
    return random_permutation_map


def target_transform(index, target, wrong_labels_map, wrong_percentage, check_set, check_verbose=False):
    rng = random.Random(index)  # Create a separate RNG seeded with the index
    if check_verbose and index in check_set:
        print(f"Index: {index}, Target: {target}")
    if rng.random() < wrong_percentage:
        return wrong_labels_map[target]
    else:
        return target


def train(args, model, device, train_loader, optimizer, epoch, target_transform_fn):
    model.train()
    for batch_idx, (indices, (data, target)) in enumerate(train_loader):
        target = torch.tensor(
            [target_transform_fn(index.item(), t.item()) for index, t in zip(indices, target)])

        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--no-mps', action='store_true', default=False,
                        help='disables macOS GPU training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    use_mps = not args.no_mps and torch.backends.mps.is_available()

    torch.manual_seed(args.seed)

    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    dataset1 = IndexedDataset(datasets.MNIST('../data', train=True, download=True,
                                             transform=transform))
    dataset2 = datasets.MNIST('../data', train=False,
                              transform=transform)

    wrong_labels_map = generate_wrong_labels_map([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    print("Wrong labels map: ", wrong_labels_map)
    check_set = set(random.sample(range(len(dataset1)), 10))
    print("Check set: ", check_set)
    wrong_percentage = 0.45
    target_transform_fn = lambda index, target: target_transform(index, target,
                                                                 wrong_labels_map=wrong_labels_map,
                                                                 wrong_percentage=wrong_percentage,
                                                                 check_set=check_set,
                                                                 check_verbose=False)

    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    model = Net().to(device)
    # optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch, target_transform_fn)
        test(model, device, test_loader)
        scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == '__main__':
    main()

    
    
#mnist_test.py --lr 0.001 

#Output
Wrong labels map:  {0: 8, 1: 7, 2: 5, 3: 9, 4: 2, 5: 3, 6: 6, 7: 1, 8: 4, 9: 0}
Check set:  {55842, 35140, 3401, 21770, 8851, 36852, 22099, 36763, 22334, 10463}
Train Epoch: 1 [0/60000 (0%)]	Loss: 2.294273
Train Epoch: 1 [640/60000 (1%)]	Loss: 2.001202
Train Epoch: 1 [1280/60000 (2%)]	Loss: 1.876791
Train Epoch: 1 [1920/60000 (3%)]	Loss: 1.590156
Train Epoch: 1 [2560/60000 (4%)]	Loss: 1.282474
Train Epoch: 1 [3200/60000 (5%)]	Loss: 1.253428
Train Epoch: 1 [3840/60000 (6%)]	Loss: 1.087605
Train Epoch: 1 [4480/60000 (7%)]	Loss: 1.362329
Train Epoch: 1 [5120/60000 (9%)]	Loss: 1.135270
Train Epoch: 1 [5760/60000 (10%)]	Loss: 0.930452
Train Epoch: 1 [6400/60000 (11%)]	Loss: 1.038844
Train Epoch: 1 [7040/60000 (12%)]	Loss: 1.077383
Train Epoch: 1 [7680/60000 (13%)]	Loss: 0.990299
Train Epoch: 1 [8320/60000 (14%)]	Loss: 0.924259
Train Epoch: 1 [8960/60000 (15%)]	Loss: 1.022265
Train Epoch: 1 [9600/60000 (16%)]	Loss: 0.846772
Train Epoch: 1 [10240/60000 (17%)]	Loss: 0.911261
Train Epoch: 1 [10880/60000 (18%)]	Loss: 0.936145
Train Epoch: 1 [11520/60000 (19%)]	Loss: 0.963351
Train Epoch: 1 [12160/60000 (20%)]	Loss: 0.814227
Train Epoch: 1 [12800/60000 (21%)]	Loss: 0.916957
Train Epoch: 1 [13440/60000 (22%)]	Loss: 0.914564
Train Epoch: 1 [14080/60000 (23%)]	Loss: 0.763393
Train Epoch: 1 [14720/60000 (25%)]	Loss: 0.744956
Train Epoch: 1 [15360/60000 (26%)]	Loss: 1.034170
Train Epoch: 1 [16000/60000 (27%)]	Loss: 0.830317
Train Epoch: 1 [16640/60000 (28%)]	Loss: 0.757520
Train Epoch: 1 [17280/60000 (29%)]	Loss: 0.763293
Train Epoch: 1 [17920/60000 (30%)]	Loss: 0.743159
Train Epoch: 1 [18560/60000 (31%)]	Loss: 0.941133
Train Epoch: 1 [19200/60000 (32%)]	Loss: 0.843691
Train Epoch: 1 [19840/60000 (33%)]	Loss: 0.792446
Train Epoch: 1 [20480/60000 (34%)]	Loss: 0.896975
Train Epoch: 1 [21120/60000 (35%)]	Loss: 0.764431
Train Epoch: 1 [21760/60000 (36%)]	Loss: 0.980849
Train Epoch: 1 [22400/60000 (37%)]	Loss: 0.788719
Train Epoch: 1 [23040/60000 (38%)]	Loss: 0.874988
Train Epoch: 1 [23680/60000 (39%)]	Loss: 0.854161
Train Epoch: 1 [24320/60000 (41%)]	Loss: 0.740625
Train Epoch: 1 [24960/60000 (42%)]	Loss: 0.825370
Train Epoch: 1 [25600/60000 (43%)]	Loss: 0.784613
Train Epoch: 1 [26240/60000 (44%)]	Loss: 0.990189
Train Epoch: 1 [26880/60000 (45%)]	Loss: 0.941862
Train Epoch: 1 [27520/60000 (46%)]	Loss: 0.715363
Train Epoch: 1 [28160/60000 (47%)]	Loss: 0.813568
Train Epoch: 1 [28800/60000 (48%)]	Loss: 0.644515
Train Epoch: 1 [29440/60000 (49%)]	Loss: 0.727208
Train Epoch: 1 [30080/60000 (50%)]	Loss: 0.781635
Train Epoch: 1 [30720/60000 (51%)]	Loss: 0.836623
Train Epoch: 1 [31360/60000 (52%)]	Loss: 0.631024
Train Epoch: 1 [32000/60000 (53%)]	Loss: 0.838560
Train Epoch: 1 [32640/60000 (54%)]	Loss: 0.741631
Train Epoch: 1 [33280/60000 (55%)]	Loss: 0.826744
Train Epoch: 1 [33920/60000 (57%)]	Loss: 0.856884
Train Epoch: 1 [34560/60000 (58%)]	Loss: 0.696886
Train Epoch: 1 [35200/60000 (59%)]	Loss: 0.884957
Train Epoch: 1 [35840/60000 (60%)]	Loss: 0.765562
Train Epoch: 1 [36480/60000 (61%)]	Loss: 0.876152
Train Epoch: 1 [37120/60000 (62%)]	Loss: 0.759348
Train Epoch: 1 [37760/60000 (63%)]	Loss: 0.745024
Train Epoch: 1 [38400/60000 (64%)]	Loss: 0.762923
Train Epoch: 1 [39040/60000 (65%)]	Loss: 0.687505
Train Epoch: 1 [39680/60000 (66%)]	Loss: 0.925046
Train Epoch: 1 [40320/60000 (67%)]	Loss: 0.863224
Train Epoch: 1 [40960/60000 (68%)]	Loss: 0.712683
Train Epoch: 1 [41600/60000 (69%)]	Loss: 0.994507
Train Epoch: 1 [42240/60000 (70%)]	Loss: 0.836363
Train Epoch: 1 [42880/60000 (71%)]	Loss: 0.681053
Train Epoch: 1 [43520/60000 (72%)]	Loss: 0.801639
Train Epoch: 1 [44160/60000 (74%)]	Loss: 0.706818
Train Epoch: 1 [44800/60000 (75%)]	Loss: 0.708225
Train Epoch: 1 [45440/60000 (76%)]	Loss: 0.712439
Train Epoch: 1 [46080/60000 (77%)]	Loss: 0.691050
Train Epoch: 1 [46720/60000 (78%)]	Loss: 0.727657
Train Epoch: 1 [47360/60000 (79%)]	Loss: 0.856804
Train Epoch: 1 [48000/60000 (80%)]	Loss: 0.748323
Train Epoch: 1 [48640/60000 (81%)]	Loss: 0.856984
Train Epoch: 1 [49280/60000 (82%)]	Loss: 0.723647
Train Epoch: 1 [49920/60000 (83%)]	Loss: 0.881680
Train Epoch: 1 [50560/60000 (84%)]	Loss: 0.684094
Train Epoch: 1 [51200/60000 (85%)]	Loss: 0.682143
Train Epoch: 1 [51840/60000 (86%)]	Loss: 0.736962
Train Epoch: 1 [52480/60000 (87%)]	Loss: 0.798493
Train Epoch: 1 [53120/60000 (88%)]	Loss: 0.692938
Train Epoch: 1 [53760/60000 (90%)]	Loss: 0.721738
Train Epoch: 1 [54400/60000 (91%)]	Loss: 0.617326
Train Epoch: 1 [55040/60000 (92%)]	Loss: 1.002632
Train Epoch: 1 [55680/60000 (93%)]	Loss: 0.900213
Train Epoch: 1 [56320/60000 (94%)]	Loss: 0.926352
Train Epoch: 1 [56960/60000 (95%)]	Loss: 0.746513
Train Epoch: 1 [57600/60000 (96%)]	Loss: 0.773066
Train Epoch: 1 [58240/60000 (97%)]	Loss: 0.824428
Train Epoch: 1 [58880/60000 (98%)]	Loss: 0.760698
Train Epoch: 1 [59520/60000 (99%)]	Loss: 0.808560

Test set: Average loss: 0.6223, Accuracy: 7244/10000 (72%)

Train Epoch: 2 [0/60000 (0%)]	Loss: 0.683797
Train Epoch: 2 [640/60000 (1%)]	Loss: 0.575442
Train Epoch: 2 [1280/60000 (2%)]	Loss: 0.838633
Train Epoch: 2 [1920/60000 (3%)]	Loss: 0.815384
Train Epoch: 2 [2560/60000 (4%)]	Loss: 0.739644
Train Epoch: 2 [3200/60000 (5%)]	Loss: 0.682403
Train Epoch: 2 [3840/60000 (6%)]	Loss: 0.745059
Train Epoch: 2 [4480/60000 (7%)]	Loss: 0.775088
Train Epoch: 2 [5120/60000 (9%)]	Loss: 0.693444
Train Epoch: 2 [5760/60000 (10%)]	Loss: 0.853091
Train Epoch: 2 [6400/60000 (11%)]	Loss: 0.741872
Train Epoch: 2 [7040/60000 (12%)]	Loss: 0.705043
Train Epoch: 2 [7680/60000 (13%)]	Loss: 0.796313
Train Epoch: 2 [8320/60000 (14%)]	Loss: 0.721148
Train Epoch: 2 [8960/60000 (15%)]	Loss: 0.753424
Train Epoch: 2 [9600/60000 (16%)]	Loss: 0.726394
Train Epoch: 2 [10240/60000 (17%)]	Loss: 0.738786
Train Epoch: 2 [10880/60000 (18%)]	Loss: 0.646009
Train Epoch: 2 [11520/60000 (19%)]	Loss: 0.761224
Train Epoch: 2 [12160/60000 (20%)]	Loss: 0.644953
Train Epoch: 2 [12800/60000 (21%)]	Loss: 0.682049
Train Epoch: 2 [13440/60000 (22%)]	Loss: 0.700257
Train Epoch: 2 [14080/60000 (23%)]	Loss: 0.621609
Train Epoch: 2 [14720/60000 (25%)]	Loss: 0.740064
Train Epoch: 2 [15360/60000 (26%)]	Loss: 0.764155
Train Epoch: 2 [16000/60000 (27%)]	Loss: 0.643151
Train Epoch: 2 [16640/60000 (28%)]	Loss: 0.798654
Train Epoch: 2 [17280/60000 (29%)]	Loss: 0.706138
Train Epoch: 2 [17920/60000 (30%)]	Loss: 0.688586
Train Epoch: 2 [18560/60000 (31%)]	Loss: 0.785688
Train Epoch: 2 [19200/60000 (32%)]	Loss: 0.639276
Train Epoch: 2 [19840/60000 (33%)]	Loss: 0.772321
Train Epoch: 2 [20480/60000 (34%)]	Loss: 0.684064
Train Epoch: 2 [21120/60000 (35%)]	Loss: 0.683842
Train Epoch: 2 [21760/60000 (36%)]	Loss: 0.706721
Train Epoch: 2 [22400/60000 (37%)]	Loss: 0.639617
Train Epoch: 2 [23040/60000 (38%)]	Loss: 0.671848
Train Epoch: 2 [23680/60000 (39%)]	Loss: 0.819904
Train Epoch: 2 [24320/60000 (41%)]	Loss: 0.625643
Train Epoch: 2 [24960/60000 (42%)]	Loss: 0.728006
Train Epoch: 2 [25600/60000 (43%)]	Loss: 0.695157
Train Epoch: 2 [26240/60000 (44%)]	Loss: 0.700835
Train Epoch: 2 [26880/60000 (45%)]	Loss: 0.729410
Train Epoch: 2 [27520/60000 (46%)]	Loss: 0.648260
Train Epoch: 2 [28160/60000 (47%)]	Loss: 0.644062
Train Epoch: 2 [28800/60000 (48%)]	Loss: 0.968753
Train Epoch: 2 [29440/60000 (49%)]	Loss: 0.692003
Train Epoch: 2 [30080/60000 (50%)]	Loss: 0.706633
Train Epoch: 2 [30720/60000 (51%)]	Loss: 0.719208
Train Epoch: 2 [31360/60000 (52%)]	Loss: 0.669756
Train Epoch: 2 [32000/60000 (53%)]	Loss: 0.664346
Train Epoch: 2 [32640/60000 (54%)]	Loss: 0.752699
Train Epoch: 2 [33280/60000 (55%)]	Loss: 0.756950
Train Epoch: 2 [33920/60000 (57%)]	Loss: 0.656988
Train Epoch: 2 [34560/60000 (58%)]	Loss: 0.716945
Train Epoch: 2 [35200/60000 (59%)]	Loss: 0.684341
Train Epoch: 2 [35840/60000 (60%)]	Loss: 0.727434
Train Epoch: 2 [36480/60000 (61%)]	Loss: 0.771401
Train Epoch: 2 [37120/60000 (62%)]	Loss: 0.791830
Train Epoch: 2 [37760/60000 (63%)]	Loss: 0.680209
Train Epoch: 2 [38400/60000 (64%)]	Loss: 0.644897
Train Epoch: 2 [39040/60000 (65%)]	Loss: 0.679021
Train Epoch: 2 [39680/60000 (66%)]	Loss: 0.716990
Train Epoch: 2 [40320/60000 (67%)]	Loss: 0.666654
Train Epoch: 2 [40960/60000 (68%)]	Loss: 0.651081
Train Epoch: 2 [41600/60000 (69%)]	Loss: 0.661147
Train Epoch: 2 [42240/60000 (70%)]	Loss: 0.710109
Train Epoch: 2 [42880/60000 (71%)]	Loss: 0.639433
Train Epoch: 2 [43520/60000 (72%)]	Loss: 0.688274
Train Epoch: 2 [44160/60000 (74%)]	Loss: 0.678582
Train Epoch: 2 [44800/60000 (75%)]	Loss: 0.721359
Train Epoch: 2 [45440/60000 (76%)]	Loss: 0.598433
Train Epoch: 2 [46080/60000 (77%)]	Loss: 0.827004
Train Epoch: 2 [46720/60000 (78%)]	Loss: 0.710028
Train Epoch: 2 [47360/60000 (79%)]	Loss: 0.691893
Train Epoch: 2 [48000/60000 (80%)]	Loss: 0.663946
Train Epoch: 2 [48640/60000 (81%)]	Loss: 0.785949
Train Epoch: 2 [49280/60000 (82%)]	Loss: 0.799669
Train Epoch: 2 [49920/60000 (83%)]	Loss: 0.637911
Train Epoch: 2 [50560/60000 (84%)]	Loss: 0.722239
Train Epoch: 2 [51200/60000 (85%)]	Loss: 0.650877
Train Epoch: 2 [51840/60000 (86%)]	Loss: 0.704657
Train Epoch: 2 [52480/60000 (87%)]	Loss: 0.652667
Train Epoch: 2 [53120/60000 (88%)]	Loss: 0.781312
Train Epoch: 2 [53760/60000 (90%)]	Loss: 0.918913
Train Epoch: 2 [54400/60000 (91%)]	Loss: 0.663945
Train Epoch: 2 [55040/60000 (92%)]	Loss: 0.664458
Train Epoch: 2 [55680/60000 (93%)]	Loss: 0.773039
Train Epoch: 2 [56320/60000 (94%)]	Loss: 0.607099
Train Epoch: 2 [56960/60000 (95%)]	Loss: 0.735135
Train Epoch: 2 [57600/60000 (96%)]	Loss: 0.702730
Train Epoch: 2 [58240/60000 (97%)]	Loss: 0.713244
Train Epoch: 2 [58880/60000 (98%)]	Loss: 0.748449
Train Epoch: 2 [59520/60000 (99%)]	Loss: 0.671776

Test set: Average loss: 0.5941, Accuracy: 8388/10000 (84%)

Train Epoch: 3 [0/60000 (0%)]	Loss: 0.762887
Train Epoch: 3 [640/60000 (1%)]	Loss: 0.651923
Train Epoch: 3 [1280/60000 (2%)]	Loss: 0.637966
Train Epoch: 3 [1920/60000 (3%)]	Loss: 0.749056
Train Epoch: 3 [2560/60000 (4%)]	Loss: 0.610169
Train Epoch: 3 [3200/60000 (5%)]	Loss: 0.706990
Train Epoch: 3 [3840/60000 (6%)]	Loss: 0.699355
Train Epoch: 3 [4480/60000 (7%)]	Loss: 0.717790
Train Epoch: 3 [5120/60000 (9%)]	Loss: 0.726413
Train Epoch: 3 [5760/60000 (10%)]	Loss: 0.600632
Train Epoch: 3 [6400/60000 (11%)]	Loss: 0.705370
Train Epoch: 3 [7040/60000 (12%)]	Loss: 0.714027
Train Epoch: 3 [7680/60000 (13%)]	Loss: 0.629736
Train Epoch: 3 [8320/60000 (14%)]	Loss: 0.661517
Train Epoch: 3 [8960/60000 (15%)]	Loss: 0.697656
Train Epoch: 3 [9600/60000 (16%)]	Loss: 0.660568
Train Epoch: 3 [10240/60000 (17%)]	Loss: 0.615648
Train Epoch: 3 [10880/60000 (18%)]	Loss: 0.698846
Train Epoch: 3 [11520/60000 (19%)]	Loss: 0.684921
Train Epoch: 3 [12160/60000 (20%)]	Loss: 0.673406
Train Epoch: 3 [12800/60000 (21%)]	Loss: 0.711251
Train Epoch: 3 [13440/60000 (22%)]	Loss: 0.636422
Train Epoch: 3 [14080/60000 (23%)]	Loss: 0.845295
Train Epoch: 3 [14720/60000 (25%)]	Loss: 0.633751
Train Epoch: 3 [15360/60000 (26%)]	Loss: 0.745765
Train Epoch: 3 [16000/60000 (27%)]	Loss: 0.858765
Train Epoch: 3 [16640/60000 (28%)]	Loss: 0.694728
Train Epoch: 3 [17280/60000 (29%)]	Loss: 0.702471
Train Epoch: 3 [17920/60000 (30%)]	Loss: 0.794016
Train Epoch: 3 [18560/60000 (31%)]	Loss: 0.700204
Train Epoch: 3 [19200/60000 (32%)]	Loss: 0.636363
Train Epoch: 3 [19840/60000 (33%)]	Loss: 0.660285
Train Epoch: 3 [20480/60000 (34%)]	Loss: 0.607656
Train Epoch: 3 [21120/60000 (35%)]	Loss: 0.697806
Train Epoch: 3 [21760/60000 (36%)]	Loss: 0.764459
Train Epoch: 3 [22400/60000 (37%)]	Loss: 0.788749
Train Epoch: 3 [23040/60000 (38%)]	Loss: 0.732966
Train Epoch: 3 [23680/60000 (39%)]	Loss: 0.659098
Train Epoch: 3 [24320/60000 (41%)]	Loss: 0.752454
Train Epoch: 3 [24960/60000 (42%)]	Loss: 0.755213
Train Epoch: 3 [25600/60000 (43%)]	Loss: 0.620229
Train Epoch: 3 [26240/60000 (44%)]	Loss: 0.677471
Train Epoch: 3 [26880/60000 (45%)]	Loss: 0.705357
Train Epoch: 3 [27520/60000 (46%)]	Loss: 0.686085
Train Epoch: 3 [28160/60000 (47%)]	Loss: 0.686143
Train Epoch: 3 [28800/60000 (48%)]	Loss: 0.719451
Train Epoch: 3 [29440/60000 (49%)]	Loss: 0.667161
Train Epoch: 3 [30080/60000 (50%)]	Loss: 0.681564
Train Epoch: 3 [30720/60000 (51%)]	Loss: 0.684567
Train Epoch: 3 [31360/60000 (52%)]	Loss: 0.655255
Train Epoch: 3 [32000/60000 (53%)]	Loss: 0.644209
Train Epoch: 3 [32640/60000 (54%)]	Loss: 0.640326
Train Epoch: 3 [33280/60000 (55%)]	Loss: 0.611766
Train Epoch: 3 [33920/60000 (57%)]	Loss: 0.614189
Train Epoch: 3 [34560/60000 (58%)]	Loss: 0.693660
Train Epoch: 3 [35200/60000 (59%)]	Loss: 0.648425
Train Epoch: 3 [35840/60000 (60%)]	Loss: 0.867421
Train Epoch: 3 [36480/60000 (61%)]	Loss: 0.668894
Train Epoch: 3 [37120/60000 (62%)]	Loss: 0.794638
Train Epoch: 3 [37760/60000 (63%)]	Loss: 0.808252
Train Epoch: 3 [38400/60000 (64%)]	Loss: 0.726869
Train Epoch: 3 [39040/60000 (65%)]	Loss: 0.759963
Train Epoch: 3 [39680/60000 (66%)]	Loss: 0.634173
Train Epoch: 3 [40320/60000 (67%)]	Loss: 0.656712
Train Epoch: 3 [40960/60000 (68%)]	Loss: 0.674716
Train Epoch: 3 [41600/60000 (69%)]	Loss: 0.702682
Train Epoch: 3 [42240/60000 (70%)]	Loss: 0.749492
Train Epoch: 3 [42880/60000 (71%)]	Loss: 0.832280
Train Epoch: 3 [43520/60000 (72%)]	Loss: 0.593578
Train Epoch: 3 [44160/60000 (74%)]	Loss: 0.616728
Train Epoch: 3 [44800/60000 (75%)]	Loss: 0.590192
Train Epoch: 3 [45440/60000 (76%)]	Loss: 0.763142
Train Epoch: 3 [46080/60000 (77%)]	Loss: 0.698444
Train Epoch: 3 [46720/60000 (78%)]	Loss: 0.692054
Train Epoch: 3 [47360/60000 (79%)]	Loss: 0.662113
Train Epoch: 3 [48000/60000 (80%)]	Loss: 0.683486
Train Epoch: 3 [48640/60000 (81%)]	Loss: 0.589330
Train Epoch: 3 [49280/60000 (82%)]	Loss: 0.670411
Train Epoch: 3 [49920/60000 (83%)]	Loss: 0.730766
Train Epoch: 3 [50560/60000 (84%)]	Loss: 0.567674
Train Epoch: 3 [51200/60000 (85%)]	Loss: 0.797317
Train Epoch: 3 [51840/60000 (86%)]	Loss: 0.621285
Train Epoch: 3 [52480/60000 (87%)]	Loss: 0.670371
Train Epoch: 3 [53120/60000 (88%)]	Loss: 0.804865
Train Epoch: 3 [53760/60000 (90%)]	Loss: 0.730104
Train Epoch: 3 [54400/60000 (91%)]	Loss: 0.713454
Train Epoch: 3 [55040/60000 (92%)]	Loss: 0.701092
Train Epoch: 3 [55680/60000 (93%)]	Loss: 0.730187
Train Epoch: 3 [56320/60000 (94%)]	Loss: 0.779162
Train Epoch: 3 [56960/60000 (95%)]	Loss: 0.683542
Train Epoch: 3 [57600/60000 (96%)]	Loss: 0.622354
Train Epoch: 3 [58240/60000 (97%)]	Loss: 0.716815
Train Epoch: 3 [58880/60000 (98%)]	Loss: 0.698626
Train Epoch: 3 [59520/60000 (99%)]	Loss: 0.684798

Test set: Average loss: 0.5936, Accuracy: 7881/10000 (79%)

Train Epoch: 4 [0/60000 (0%)]	Loss: 0.633560
Train Epoch: 4 [640/60000 (1%)]	Loss: 0.590945
Train Epoch: 4 [1280/60000 (2%)]	Loss: 0.730103
Train Epoch: 4 [1920/60000 (3%)]	Loss: 0.652385
Train Epoch: 4 [2560/60000 (4%)]	Loss: 0.672824
Train Epoch: 4 [3200/60000 (5%)]	Loss: 0.643391
Train Epoch: 4 [3840/60000 (6%)]	Loss: 0.682402
Train Epoch: 4 [4480/60000 (7%)]	Loss: 0.693219
Train Epoch: 4 [5120/60000 (9%)]	Loss: 0.649202
Train Epoch: 4 [5760/60000 (10%)]	Loss: 0.579442
Train Epoch: 4 [6400/60000 (11%)]	Loss: 0.637233
Train Epoch: 4 [7040/60000 (12%)]	Loss: 0.658160
Train Epoch: 4 [7680/60000 (13%)]	Loss: 0.636211
Train Epoch: 4 [8320/60000 (14%)]	Loss: 0.667626
Train Epoch: 4 [8960/60000 (15%)]	Loss: 0.681543
Train Epoch: 4 [9600/60000 (16%)]	Loss: 0.664423
Train Epoch: 4 [10240/60000 (17%)]	Loss: 0.584629
Train Epoch: 4 [10880/60000 (18%)]	Loss: 0.727324
Train Epoch: 4 [11520/60000 (19%)]	Loss: 0.642825
Train Epoch: 4 [12160/60000 (20%)]	Loss: 0.693080
Train Epoch: 4 [12800/60000 (21%)]	Loss: 0.695518
Train Epoch: 4 [13440/60000 (22%)]	Loss: 0.844392
Train Epoch: 4 [14080/60000 (23%)]	Loss: 0.686393
Train Epoch: 4 [14720/60000 (25%)]	Loss: 0.637060
Train Epoch: 4 [15360/60000 (26%)]	Loss: 0.627431
Train Epoch: 4 [16000/60000 (27%)]	Loss: 0.651379
Train Epoch: 4 [16640/60000 (28%)]	Loss: 0.623133
Train Epoch: 4 [17280/60000 (29%)]	Loss: 0.651633
Train Epoch: 4 [17920/60000 (30%)]	Loss: 0.712870
Train Epoch: 4 [18560/60000 (31%)]	Loss: 0.649325
Train Epoch: 4 [19200/60000 (32%)]	Loss: 0.655255
Train Epoch: 4 [19840/60000 (33%)]	Loss: 0.631121
Train Epoch: 4 [20480/60000 (34%)]	Loss: 0.692468
Train Epoch: 4 [21120/60000 (35%)]	Loss: 0.669448
Train Epoch: 4 [21760/60000 (36%)]	Loss: 0.612344
Train Epoch: 4 [22400/60000 (37%)]	Loss: 0.696051
Train Epoch: 4 [23040/60000 (38%)]	Loss: 0.727948
Train Epoch: 4 [23680/60000 (39%)]	Loss: 0.705175
Train Epoch: 4 [24320/60000 (41%)]	Loss: 0.592122
Train Epoch: 4 [24960/60000 (42%)]	Loss: 0.600667
Train Epoch: 4 [25600/60000 (43%)]	Loss: 0.691833
Train Epoch: 4 [26240/60000 (44%)]	Loss: 0.645699
Train Epoch: 4 [26880/60000 (45%)]	Loss: 0.652517
Train Epoch: 4 [27520/60000 (46%)]	Loss: 0.837824
Train Epoch: 4 [28160/60000 (47%)]	Loss: 0.646894
Train Epoch: 4 [28800/60000 (48%)]	Loss: 0.682552
Train Epoch: 4 [29440/60000 (49%)]	Loss: 0.604886
Train Epoch: 4 [30080/60000 (50%)]	Loss: 0.654284
Train Epoch: 4 [30720/60000 (51%)]	Loss: 0.717625
Train Epoch: 4 [31360/60000 (52%)]	Loss: 0.641939
Train Epoch: 4 [32000/60000 (53%)]	Loss: 0.633307
Train Epoch: 4 [32640/60000 (54%)]	Loss: 0.606494
Train Epoch: 4 [33280/60000 (55%)]	Loss: 0.674524
Train Epoch: 4 [33920/60000 (57%)]	Loss: 0.693996
Train Epoch: 4 [34560/60000 (58%)]	Loss: 0.660512
Train Epoch: 4 [35200/60000 (59%)]	Loss: 0.699750
Train Epoch: 4 [35840/60000 (60%)]	Loss: 0.640560
Train Epoch: 4 [36480/60000 (61%)]	Loss: 0.658035
Train Epoch: 4 [37120/60000 (62%)]	Loss: 0.722309
Train Epoch: 4 [37760/60000 (63%)]	Loss: 0.691034
Train Epoch: 4 [38400/60000 (64%)]	Loss: 0.690797
Train Epoch: 4 [39040/60000 (65%)]	Loss: 0.793144
Train Epoch: 4 [39680/60000 (66%)]	Loss: 0.684306
Train Epoch: 4 [40320/60000 (67%)]	Loss: 0.653133
Train Epoch: 4 [40960/60000 (68%)]	Loss: 0.685833
Train Epoch: 4 [41600/60000 (69%)]	Loss: 0.646209
Train Epoch: 4 [42240/60000 (70%)]	Loss: 0.695993
Train Epoch: 4 [42880/60000 (71%)]	Loss: 0.648641
Train Epoch: 4 [43520/60000 (72%)]	Loss: 0.660974
Train Epoch: 4 [44160/60000 (74%)]	Loss: 0.597732
Train Epoch: 4 [44800/60000 (75%)]	Loss: 0.599662
Train Epoch: 4 [45440/60000 (76%)]	Loss: 0.578761
Train Epoch: 4 [46080/60000 (77%)]	Loss: 0.695552
Train Epoch: 4 [46720/60000 (78%)]	Loss: 0.696418
Train Epoch: 4 [47360/60000 (79%)]	Loss: 0.629750
Train Epoch: 4 [48000/60000 (80%)]	Loss: 0.693698
Train Epoch: 4 [48640/60000 (81%)]	Loss: 0.677091
Train Epoch: 4 [49280/60000 (82%)]	Loss: 0.661487
Train Epoch: 4 [49920/60000 (83%)]	Loss: 0.640886
Train Epoch: 4 [50560/60000 (84%)]	Loss: 0.692319
Train Epoch: 4 [51200/60000 (85%)]	Loss: 0.674132
Train Epoch: 4 [51840/60000 (86%)]	Loss: 0.607334
Train Epoch: 4 [52480/60000 (87%)]	Loss: 0.622987
Train Epoch: 4 [53120/60000 (88%)]	Loss: 0.707608
Train Epoch: 4 [53760/60000 (90%)]	Loss: 0.672292
Train Epoch: 4 [54400/60000 (91%)]	Loss: 0.624886
Train Epoch: 4 [55040/60000 (92%)]	Loss: 0.652276
Train Epoch: 4 [55680/60000 (93%)]	Loss: 0.619871
Train Epoch: 4 [56320/60000 (94%)]	Loss: 0.683636
Train Epoch: 4 [56960/60000 (95%)]	Loss: 0.630637
Train Epoch: 4 [57600/60000 (96%)]	Loss: 0.652258
Train Epoch: 4 [58240/60000 (97%)]	Loss: 0.694576
Train Epoch: 4 [58880/60000 (98%)]	Loss: 0.706668
Train Epoch: 4 [59520/60000 (99%)]	Loss: 0.649582

Test set: Average loss: 0.5840, Accuracy: 7952/10000 (80%)

Train Epoch: 5 [0/60000 (0%)]	Loss: 0.694904
Train Epoch: 5 [640/60000 (1%)]	Loss: 0.556516
Train Epoch: 5 [1280/60000 (2%)]	Loss: 0.646970
Train Epoch: 5 [1920/60000 (3%)]	Loss: 0.590954
Train Epoch: 5 [2560/60000 (4%)]	Loss: 0.670378
Train Epoch: 5 [3200/60000 (5%)]	Loss: 0.691817
Train Epoch: 5 [3840/60000 (6%)]	Loss: 0.583322
Train Epoch: 5 [4480/60000 (7%)]	Loss: 0.692881
Train Epoch: 5 [5120/60000 (9%)]	Loss: 0.790744
Train Epoch: 5 [5760/60000 (10%)]	Loss: 0.662523
Train Epoch: 5 [6400/60000 (11%)]	Loss: 0.576897
Train Epoch: 5 [7040/60000 (12%)]	Loss: 0.676493
Train Epoch: 5 [7680/60000 (13%)]	Loss: 0.647071
Train Epoch: 5 [8320/60000 (14%)]	Loss: 0.584605
Train Epoch: 5 [8960/60000 (15%)]	Loss: 0.732672
Train Epoch: 5 [9600/60000 (16%)]	Loss: 0.674778
Train Epoch: 5 [10240/60000 (17%)]	Loss: 0.648317
Train Epoch: 5 [10880/60000 (18%)]	Loss: 0.707524
Train Epoch: 5 [11520/60000 (19%)]	Loss: 0.719512
Train Epoch: 5 [12160/60000 (20%)]	Loss: 0.655322
Train Epoch: 5 [12800/60000 (21%)]	Loss: 0.652121
Train Epoch: 5 [13440/60000 (22%)]	Loss: 0.654095
Train Epoch: 5 [14080/60000 (23%)]	Loss: 0.635383
Train Epoch: 5 [14720/60000 (25%)]	Loss: 0.601772
Train Epoch: 5 [15360/60000 (26%)]	Loss: 0.624785
Train Epoch: 5 [16000/60000 (27%)]	Loss: 0.624932
Train Epoch: 5 [16640/60000 (28%)]	Loss: 0.679938
Train Epoch: 5 [17280/60000 (29%)]	Loss: 0.636909
Train Epoch: 5 [17920/60000 (30%)]	Loss: 0.654622
Train Epoch: 5 [18560/60000 (31%)]	Loss: 0.641555
Train Epoch: 5 [19200/60000 (32%)]	Loss: 0.581588
Train Epoch: 5 [19840/60000 (33%)]	Loss: 0.660136
Train Epoch: 5 [20480/60000 (34%)]	Loss: 0.714150
Train Epoch: 5 [21120/60000 (35%)]	Loss: 0.645149
Train Epoch: 5 [21760/60000 (36%)]	Loss: 0.624260
Train Epoch: 5 [22400/60000 (37%)]	Loss: 0.700868
Train Epoch: 5 [23040/60000 (38%)]	Loss: 0.732507
Train Epoch: 5 [23680/60000 (39%)]	Loss: 0.599528
Train Epoch: 5 [24320/60000 (41%)]	Loss: 0.606997
Train Epoch: 5 [24960/60000 (42%)]	Loss: 0.645488
Train Epoch: 5 [25600/60000 (43%)]	Loss: 0.668090
Train Epoch: 5 [26240/60000 (44%)]	Loss: 0.604686
Train Epoch: 5 [26880/60000 (45%)]	Loss: 0.707998
Train Epoch: 5 [27520/60000 (46%)]	Loss: 0.678529
Train Epoch: 5 [28160/60000 (47%)]	Loss: 0.697898
Train Epoch: 5 [28800/60000 (48%)]	Loss: 0.669829
Train Epoch: 5 [29440/60000 (49%)]	Loss: 0.693117
Train Epoch: 5 [30080/60000 (50%)]	Loss: 0.707085
Train Epoch: 5 [30720/60000 (51%)]	Loss: 0.580006
Train Epoch: 5 [31360/60000 (52%)]	Loss: 0.648430
Train Epoch: 5 [32000/60000 (53%)]	Loss: 0.739101
Train Epoch: 5 [32640/60000 (54%)]	Loss: 0.596055
Train Epoch: 5 [33280/60000 (55%)]	Loss: 0.660920
Train Epoch: 5 [33920/60000 (57%)]	Loss: 0.678416
Train Epoch: 5 [34560/60000 (58%)]	Loss: 0.664794
Train Epoch: 5 [35200/60000 (59%)]	Loss: 0.711743
Train Epoch: 5 [35840/60000 (60%)]	Loss: 0.630791
Train Epoch: 5 [36480/60000 (61%)]	Loss: 0.679713
Train Epoch: 5 [37120/60000 (62%)]	Loss: 0.610340
Train Epoch: 5 [37760/60000 (63%)]	Loss: 0.571690
Train Epoch: 5 [38400/60000 (64%)]	Loss: 0.686173
Train Epoch: 5 [39040/60000 (65%)]	Loss: 0.710024
Train Epoch: 5 [39680/60000 (66%)]	Loss: 0.682221
Train Epoch: 5 [40320/60000 (67%)]	Loss: 0.681574
Train Epoch: 5 [40960/60000 (68%)]	Loss: 0.655876
Train Epoch: 5 [41600/60000 (69%)]	Loss: 0.580624
Train Epoch: 5 [42240/60000 (70%)]	Loss: 0.650184
Train Epoch: 5 [42880/60000 (71%)]	Loss: 0.644726
Train Epoch: 5 [43520/60000 (72%)]	Loss: 0.675460
Train Epoch: 5 [44160/60000 (74%)]	Loss: 0.703193
Train Epoch: 5 [44800/60000 (75%)]	Loss: 0.583968
Train Epoch: 5 [45440/60000 (76%)]	Loss: 0.652190
Train Epoch: 5 [46080/60000 (77%)]	Loss: 0.598000
Train Epoch: 5 [46720/60000 (78%)]	Loss: 0.617132
Train Epoch: 5 [47360/60000 (79%)]	Loss: 0.639652
Train Epoch: 5 [48000/60000 (80%)]	Loss: 0.658982
Train Epoch: 5 [48640/60000 (81%)]	Loss: 0.752916
Train Epoch: 5 [49280/60000 (82%)]	Loss: 0.769675
Train Epoch: 5 [49920/60000 (83%)]	Loss: 0.611923
Train Epoch: 5 [50560/60000 (84%)]	Loss: 0.675386
Train Epoch: 5 [51200/60000 (85%)]	Loss: 0.640931
Train Epoch: 5 [51840/60000 (86%)]	Loss: 0.664426
Train Epoch: 5 [52480/60000 (87%)]	Loss: 0.739226
Train Epoch: 5 [53120/60000 (88%)]	Loss: 0.691570
Train Epoch: 5 [53760/60000 (90%)]	Loss: 0.605744
Train Epoch: 5 [54400/60000 (91%)]	Loss: 0.692352
Train Epoch: 5 [55040/60000 (92%)]	Loss: 0.648317
Train Epoch: 5 [55680/60000 (93%)]	Loss: 0.652816
Train Epoch: 5 [56320/60000 (94%)]	Loss: 0.676496
Train Epoch: 5 [56960/60000 (95%)]	Loss: 0.585927
Train Epoch: 5 [57600/60000 (96%)]	Loss: 0.723921
Train Epoch: 5 [58240/60000 (97%)]	Loss: 0.678542
Train Epoch: 5 [58880/60000 (98%)]	Loss: 0.700499
Train Epoch: 5 [59520/60000 (99%)]	Loss: 0.616523

Test set: Average loss: 0.5811, Accuracy: 9181/10000 (92%)

Train Epoch: 6 [0/60000 (0%)]	Loss: 0.654128
Train Epoch: 6 [640/60000 (1%)]	Loss: 0.653104
Train Epoch: 6 [1280/60000 (2%)]	Loss: 0.649406
Train Epoch: 6 [1920/60000 (3%)]	Loss: 0.679362
Train Epoch: 6 [2560/60000 (4%)]	Loss: 0.642382
Train Epoch: 6 [3200/60000 (5%)]	Loss: 0.625384
Train Epoch: 6 [3840/60000 (6%)]	Loss: 0.663255
Train Epoch: 6 [4480/60000 (7%)]	Loss: 0.610145
Train Epoch: 6 [5120/60000 (9%)]	Loss: 0.609830
Train Epoch: 6 [5760/60000 (10%)]	Loss: 0.646264
Train Epoch: 6 [6400/60000 (11%)]	Loss: 0.689587
Train Epoch: 6 [7040/60000 (12%)]	Loss: 0.562047
Train Epoch: 6 [7680/60000 (13%)]	Loss: 0.634758
Train Epoch: 6 [8320/60000 (14%)]	Loss: 0.639022
Train Epoch: 6 [8960/60000 (15%)]	Loss: 0.621170
Train Epoch: 6 [9600/60000 (16%)]	Loss: 0.657986
Train Epoch: 6 [10240/60000 (17%)]	Loss: 0.667871
Train Epoch: 6 [10880/60000 (18%)]	Loss: 0.682338
Train Epoch: 6 [11520/60000 (19%)]	Loss: 0.565071
Train Epoch: 6 [12160/60000 (20%)]	Loss: 0.663739
Train Epoch: 6 [12800/60000 (21%)]	Loss: 0.665273
Train Epoch: 6 [13440/60000 (22%)]	Loss: 0.658320
Train Epoch: 6 [14080/60000 (23%)]	Loss: 0.763018
Train Epoch: 6 [14720/60000 (25%)]	Loss: 0.575847
Train Epoch: 6 [15360/60000 (26%)]	Loss: 0.750703
Train Epoch: 6 [16000/60000 (27%)]	Loss: 0.679738
Train Epoch: 6 [16640/60000 (28%)]	Loss: 0.663728
Train Epoch: 6 [17280/60000 (29%)]	Loss: 0.622766
Train Epoch: 6 [17920/60000 (30%)]	Loss: 0.743386
Train Epoch: 6 [18560/60000 (31%)]	Loss: 0.682149
Train Epoch: 6 [19200/60000 (32%)]	Loss: 0.724913
Train Epoch: 6 [19840/60000 (33%)]	Loss: 0.661800
Train Epoch: 6 [20480/60000 (34%)]	Loss: 0.721969
Train Epoch: 6 [21120/60000 (35%)]	Loss: 0.615884
Train Epoch: 6 [21760/60000 (36%)]	Loss: 0.645388
Train Epoch: 6 [22400/60000 (37%)]	Loss: 0.628495
Train Epoch: 6 [23040/60000 (38%)]	Loss: 0.666057
Train Epoch: 6 [23680/60000 (39%)]	Loss: 0.614573
Train Epoch: 6 [24320/60000 (41%)]	Loss: 0.671859
Train Epoch: 6 [24960/60000 (42%)]	Loss: 0.657372
Train Epoch: 6 [25600/60000 (43%)]	Loss: 0.647831
Train Epoch: 6 [26240/60000 (44%)]	Loss: 0.619415
Train Epoch: 6 [26880/60000 (45%)]	Loss: 0.684586
Train Epoch: 6 [27520/60000 (46%)]	Loss: 0.713308
Train Epoch: 6 [28160/60000 (47%)]	Loss: 0.659957
Train Epoch: 6 [28800/60000 (48%)]	Loss: 0.619229
Train Epoch: 6 [29440/60000 (49%)]	Loss: 0.759974
Train Epoch: 6 [30080/60000 (50%)]	Loss: 0.668031
Train Epoch: 6 [30720/60000 (51%)]	Loss: 0.674847
Train Epoch: 6 [31360/60000 (52%)]	Loss: 0.657697
Train Epoch: 6 [32000/60000 (53%)]	Loss: 0.705777
Train Epoch: 6 [32640/60000 (54%)]	Loss: 0.764480
Train Epoch: 6 [33280/60000 (55%)]	Loss: 0.634282
Train Epoch: 6 [33920/60000 (57%)]	Loss: 0.643772
Train Epoch: 6 [34560/60000 (58%)]	Loss: 0.722020
Train Epoch: 6 [35200/60000 (59%)]	Loss: 0.670247
Train Epoch: 6 [35840/60000 (60%)]	Loss: 0.632902
Train Epoch: 6 [36480/60000 (61%)]	Loss: 0.623903
Train Epoch: 6 [37120/60000 (62%)]	Loss: 0.608752
Train Epoch: 6 [37760/60000 (63%)]	Loss: 0.594475
Train Epoch: 6 [38400/60000 (64%)]	Loss: 0.598305
Train Epoch: 6 [39040/60000 (65%)]	Loss: 0.635385
Train Epoch: 6 [39680/60000 (66%)]	Loss: 0.707765
Train Epoch: 6 [40320/60000 (67%)]	Loss: 0.652199
Train Epoch: 6 [40960/60000 (68%)]	Loss: 0.616084
Train Epoch: 6 [41600/60000 (69%)]	Loss: 0.712726
Train Epoch: 6 [42240/60000 (70%)]	Loss: 0.576263
Train Epoch: 6 [42880/60000 (71%)]	Loss: 0.778304
Train Epoch: 6 [43520/60000 (72%)]	Loss: 0.650040
Train Epoch: 6 [44160/60000 (74%)]	Loss: 0.613513
Train Epoch: 6 [44800/60000 (75%)]	Loss: 0.641399
Train Epoch: 6 [45440/60000 (76%)]	Loss: 0.612501
Train Epoch: 6 [46080/60000 (77%)]	Loss: 0.699003
Train Epoch: 6 [46720/60000 (78%)]	Loss: 0.668710
Train Epoch: 6 [47360/60000 (79%)]	Loss: 0.739422
Train Epoch: 6 [48000/60000 (80%)]	Loss: 0.642822
Train Epoch: 6 [48640/60000 (81%)]	Loss: 0.568534
Train Epoch: 6 [49280/60000 (82%)]	Loss: 0.638053
Train Epoch: 6 [49920/60000 (83%)]	Loss: 0.601673
Train Epoch: 6 [50560/60000 (84%)]	Loss: 0.622601
Train Epoch: 6 [51200/60000 (85%)]	Loss: 0.643455
Train Epoch: 6 [51840/60000 (86%)]	Loss: 0.633555
Train Epoch: 6 [52480/60000 (87%)]	Loss: 0.670455
Train Epoch: 6 [53120/60000 (88%)]	Loss: 0.692142
Train Epoch: 6 [53760/60000 (90%)]	Loss: 0.597745
Train Epoch: 6 [54400/60000 (91%)]	Loss: 0.626385
Train Epoch: 6 [55040/60000 (92%)]	Loss: 0.667085
Train Epoch: 6 [55680/60000 (93%)]	Loss: 0.611081
Train Epoch: 6 [56320/60000 (94%)]	Loss: 0.718847
Train Epoch: 6 [56960/60000 (95%)]	Loss: 0.667910
Train Epoch: 6 [57600/60000 (96%)]	Loss: 0.702350
Train Epoch: 6 [58240/60000 (97%)]	Loss: 0.673018
Train Epoch: 6 [58880/60000 (98%)]	Loss: 0.554907
Train Epoch: 6 [59520/60000 (99%)]	Loss: 0.633087

Test set: Average loss: 0.5988, Accuracy: 7785/10000 (78%)

Train Epoch: 7 [0/60000 (0%)]	Loss: 0.638511
Train Epoch: 7 [640/60000 (1%)]	Loss: 0.770964
Train Epoch: 7 [1280/60000 (2%)]	Loss: 0.642032
Train Epoch: 7 [1920/60000 (3%)]	Loss: 0.609628
Train Epoch: 7 [2560/60000 (4%)]	Loss: 0.686847
Train Epoch: 7 [3200/60000 (5%)]	Loss: 0.671614
Train Epoch: 7 [3840/60000 (6%)]	Loss: 0.660754
Train Epoch: 7 [4480/60000 (7%)]	Loss: 0.637455
Train Epoch: 7 [5120/60000 (9%)]	Loss: 0.636501
Train Epoch: 7 [5760/60000 (10%)]	Loss: 0.675870
Train Epoch: 7 [6400/60000 (11%)]	Loss: 0.698564
Train Epoch: 7 [7040/60000 (12%)]	Loss: 0.650364
Train Epoch: 7 [7680/60000 (13%)]	Loss: 0.625724
Train Epoch: 7 [8320/60000 (14%)]	Loss: 0.646401
Train Epoch: 7 [8960/60000 (15%)]	Loss: 0.635856
Train Epoch: 7 [9600/60000 (16%)]	Loss: 0.607684
Train Epoch: 7 [10240/60000 (17%)]	Loss: 0.678227
Train Epoch: 7 [10880/60000 (18%)]	Loss: 0.539911
Train Epoch: 7 [11520/60000 (19%)]	Loss: 0.659084
Train Epoch: 7 [12160/60000 (20%)]	Loss: 0.690385
Train Epoch: 7 [12800/60000 (21%)]	Loss: 0.650605
Train Epoch: 7 [13440/60000 (22%)]	Loss: 0.628566
Train Epoch: 7 [14080/60000 (23%)]	Loss: 0.670758
Train Epoch: 7 [14720/60000 (25%)]	Loss: 0.643432
Train Epoch: 7 [15360/60000 (26%)]	Loss: 0.620204
Train Epoch: 7 [16000/60000 (27%)]	Loss: 0.569616
Train Epoch: 7 [16640/60000 (28%)]	Loss: 0.567563
Train Epoch: 7 [17280/60000 (29%)]	Loss: 0.636390
Train Epoch: 7 [17920/60000 (30%)]	Loss: 0.695038
Train Epoch: 7 [18560/60000 (31%)]	Loss: 0.615476
Train Epoch: 7 [19200/60000 (32%)]	Loss: 0.668458
Train Epoch: 7 [19840/60000 (33%)]	Loss: 0.676934
Train Epoch: 7 [20480/60000 (34%)]	Loss: 0.570832
Train Epoch: 7 [21120/60000 (35%)]	Loss: 0.686618
Train Epoch: 7 [21760/60000 (36%)]	Loss: 0.645376
Train Epoch: 7 [22400/60000 (37%)]	Loss: 0.639257
Train Epoch: 7 [23040/60000 (38%)]	Loss: 0.659323
Train Epoch: 7 [23680/60000 (39%)]	Loss: 0.653902
Train Epoch: 7 [24320/60000 (41%)]	Loss: 0.666504
Train Epoch: 7 [24960/60000 (42%)]	Loss: 0.735545
Train Epoch: 7 [25600/60000 (43%)]	Loss: 0.648708
Train Epoch: 7 [26240/60000 (44%)]	Loss: 0.675432
Train Epoch: 7 [26880/60000 (45%)]	Loss: 0.670157
Train Epoch: 7 [27520/60000 (46%)]	Loss: 0.659839
Train Epoch: 7 [28160/60000 (47%)]	Loss: 0.610401
Train Epoch: 7 [28800/60000 (48%)]	Loss: 0.670155
Train Epoch: 7 [29440/60000 (49%)]	Loss: 0.627349
Train Epoch: 7 [30080/60000 (50%)]	Loss: 0.727300
Train Epoch: 7 [30720/60000 (51%)]	Loss: 0.602125
Train Epoch: 7 [31360/60000 (52%)]	Loss: 0.588911
Train Epoch: 7 [32000/60000 (53%)]	Loss: 0.630785
Train Epoch: 7 [32640/60000 (54%)]	Loss: 0.686547
Train Epoch: 7 [33280/60000 (55%)]	Loss: 0.620025
Train Epoch: 7 [33920/60000 (57%)]	Loss: 0.594506
Train Epoch: 7 [34560/60000 (58%)]	Loss: 0.627090
Train Epoch: 7 [35200/60000 (59%)]	Loss: 0.622177
Train Epoch: 7 [35840/60000 (60%)]	Loss: 0.654256
Train Epoch: 7 [36480/60000 (61%)]	Loss: 0.679882
Train Epoch: 7 [37120/60000 (62%)]	Loss: 0.644377
Train Epoch: 7 [37760/60000 (63%)]	Loss: 0.661460
Train Epoch: 7 [38400/60000 (64%)]	Loss: 0.604120
Train Epoch: 7 [39040/60000 (65%)]	Loss: 0.611150
Train Epoch: 7 [39680/60000 (66%)]	Loss: 0.662021
Train Epoch: 7 [40320/60000 (67%)]	Loss: 0.716381
Train Epoch: 7 [40960/60000 (68%)]	Loss: 0.676699
Train Epoch: 7 [41600/60000 (69%)]	Loss: 0.615019
Train Epoch: 7 [42240/60000 (70%)]	Loss: 0.709563
Train Epoch: 7 [42880/60000 (71%)]	Loss: 0.655090
Train Epoch: 7 [43520/60000 (72%)]	Loss: 0.669161
Train Epoch: 7 [44160/60000 (74%)]	Loss: 0.682726
Train Epoch: 7 [44800/60000 (75%)]	Loss: 0.603819
Train Epoch: 7 [45440/60000 (76%)]	Loss: 0.684372
Train Epoch: 7 [46080/60000 (77%)]	Loss: 0.590019
Train Epoch: 7 [46720/60000 (78%)]	Loss: 0.625916
Train Epoch: 7 [47360/60000 (79%)]	Loss: 0.694461
Train Epoch: 7 [48000/60000 (80%)]	Loss: 0.649051
Train Epoch: 7 [48640/60000 (81%)]	Loss: 0.668317
Train Epoch: 7 [49280/60000 (82%)]	Loss: 0.722997
Train Epoch: 7 [49920/60000 (83%)]	Loss: 0.594326
Train Epoch: 7 [50560/60000 (84%)]	Loss: 0.651809
Train Epoch: 7 [51200/60000 (85%)]	Loss: 0.682901
Train Epoch: 7 [51840/60000 (86%)]	Loss: 0.632672
Train Epoch: 7 [52480/60000 (87%)]	Loss: 0.625150
Train Epoch: 7 [53120/60000 (88%)]	Loss: 0.658245
Train Epoch: 7 [53760/60000 (90%)]	Loss: 0.675941
Train Epoch: 7 [54400/60000 (91%)]	Loss: 0.614779
Train Epoch: 7 [55040/60000 (92%)]	Loss: 0.597100
Train Epoch: 7 [55680/60000 (93%)]	Loss: 0.592532
Train Epoch: 7 [56320/60000 (94%)]	Loss: 0.647002
Train Epoch: 7 [56960/60000 (95%)]	Loss: 0.629948
Train Epoch: 7 [57600/60000 (96%)]	Loss: 0.633389
Train Epoch: 7 [58240/60000 (97%)]	Loss: 0.639832
Train Epoch: 7 [58880/60000 (98%)]	Loss: 0.684692
Train Epoch: 7 [59520/60000 (99%)]	Loss: 0.594449

Test set: Average loss: 0.5903, Accuracy: 8693/10000 (87%)

Train Epoch: 8 [0/60000 (0%)]	Loss: 0.777509
Train Epoch: 8 [640/60000 (1%)]	Loss: 0.620377
Train Epoch: 8 [1280/60000 (2%)]	Loss: 0.721072
Train Epoch: 8 [1920/60000 (3%)]	Loss: 0.593135
Train Epoch: 8 [2560/60000 (4%)]	Loss: 0.644301
Train Epoch: 8 [3200/60000 (5%)]	Loss: 0.674137
Train Epoch: 8 [3840/60000 (6%)]	Loss: 0.588645
Train Epoch: 8 [4480/60000 (7%)]	Loss: 0.644431
Train Epoch: 8 [5120/60000 (9%)]	Loss: 0.620951
Train Epoch: 8 [5760/60000 (10%)]	Loss: 0.661020
Train Epoch: 8 [6400/60000 (11%)]	Loss: 0.626379
Train Epoch: 8 [7040/60000 (12%)]	Loss: 0.677157
Train Epoch: 8 [7680/60000 (13%)]	Loss: 0.570307
Train Epoch: 8 [8320/60000 (14%)]	Loss: 0.645728
Train Epoch: 8 [8960/60000 (15%)]	Loss: 0.673515
Train Epoch: 8 [9600/60000 (16%)]	Loss: 0.620565
Train Epoch: 8 [10240/60000 (17%)]	Loss: 0.616895
Train Epoch: 8 [10880/60000 (18%)]	Loss: 0.605675
Train Epoch: 8 [11520/60000 (19%)]	Loss: 0.676222
Train Epoch: 8 [12160/60000 (20%)]	Loss: 0.569044
Train Epoch: 8 [12800/60000 (21%)]	Loss: 0.600319
Train Epoch: 8 [13440/60000 (22%)]	Loss: 0.688003
Train Epoch: 8 [14080/60000 (23%)]	Loss: 0.680002
Train Epoch: 8 [14720/60000 (25%)]	Loss: 0.700500
Train Epoch: 8 [15360/60000 (26%)]	Loss: 0.636798
Train Epoch: 8 [16000/60000 (27%)]	Loss: 0.653512
Train Epoch: 8 [16640/60000 (28%)]	Loss: 0.637320
Train Epoch: 8 [17280/60000 (29%)]	Loss: 0.691026
Train Epoch: 8 [17920/60000 (30%)]	Loss: 0.567334
Train Epoch: 8 [18560/60000 (31%)]	Loss: 0.626917
Train Epoch: 8 [19200/60000 (32%)]	Loss: 0.631692
Train Epoch: 8 [19840/60000 (33%)]	Loss: 0.599237
Train Epoch: 8 [20480/60000 (34%)]	Loss: 0.647826
Train Epoch: 8 [21120/60000 (35%)]	Loss: 0.605999
Train Epoch: 8 [21760/60000 (36%)]	Loss: 0.640755
Train Epoch: 8 [22400/60000 (37%)]	Loss: 0.584815
Train Epoch: 8 [23040/60000 (38%)]	Loss: 0.568436
Train Epoch: 8 [23680/60000 (39%)]	Loss: 0.605392
Train Epoch: 8 [24320/60000 (41%)]	Loss: 0.636647
Train Epoch: 8 [24960/60000 (42%)]	Loss: 0.617734
Train Epoch: 8 [25600/60000 (43%)]	Loss: 0.586282
Train Epoch: 8 [26240/60000 (44%)]	Loss: 0.581383
Train Epoch: 8 [26880/60000 (45%)]	Loss: 0.585729
Train Epoch: 8 [27520/60000 (46%)]	Loss: 0.625881
Train Epoch: 8 [28160/60000 (47%)]	Loss: 0.708675
Train Epoch: 8 [28800/60000 (48%)]	Loss: 0.655747
Train Epoch: 8 [29440/60000 (49%)]	Loss: 0.631909
Train Epoch: 8 [30080/60000 (50%)]	Loss: 0.646663
Train Epoch: 8 [30720/60000 (51%)]	Loss: 0.618345
Train Epoch: 8 [31360/60000 (52%)]	Loss: 0.661836
Train Epoch: 8 [32000/60000 (53%)]	Loss: 0.617299
Train Epoch: 8 [32640/60000 (54%)]	Loss: 0.640105
Train Epoch: 8 [33280/60000 (55%)]	Loss: 0.649128
Train Epoch: 8 [33920/60000 (57%)]	Loss: 0.672067
Train Epoch: 8 [34560/60000 (58%)]	Loss: 0.633127
Train Epoch: 8 [35200/60000 (59%)]	Loss: 0.642266
Train Epoch: 8 [35840/60000 (60%)]	Loss: 0.598697
Train Epoch: 8 [36480/60000 (61%)]	Loss: 0.621746
Train Epoch: 8 [37120/60000 (62%)]	Loss: 0.632653
Train Epoch: 8 [37760/60000 (63%)]	Loss: 0.674431
Train Epoch: 8 [38400/60000 (64%)]	Loss: 0.672954
Train Epoch: 8 [39040/60000 (65%)]	Loss: 0.640914
Train Epoch: 8 [39680/60000 (66%)]	Loss: 0.612006
Train Epoch: 8 [40320/60000 (67%)]	Loss: 0.683859
Train Epoch: 8 [40960/60000 (68%)]	Loss: 0.655334
Train Epoch: 8 [41600/60000 (69%)]	Loss: 0.633945
Train Epoch: 8 [42240/60000 (70%)]	Loss: 0.832454
Train Epoch: 8 [42880/60000 (71%)]	Loss: 0.547897
Train Epoch: 8 [43520/60000 (72%)]	Loss: 0.633969
Train Epoch: 8 [44160/60000 (74%)]	Loss: 0.647165
Train Epoch: 8 [44800/60000 (75%)]	Loss: 0.628527
Train Epoch: 8 [45440/60000 (76%)]	Loss: 0.707152
Train Epoch: 8 [46080/60000 (77%)]	Loss: 0.629156
Train Epoch: 8 [46720/60000 (78%)]	Loss: 0.593440
Train Epoch: 8 [47360/60000 (79%)]	Loss: 0.633723
Train Epoch: 8 [48000/60000 (80%)]	Loss: 0.579859
Train Epoch: 8 [48640/60000 (81%)]	Loss: 0.559021
Train Epoch: 8 [49280/60000 (82%)]	Loss: 0.676496
Train Epoch: 8 [49920/60000 (83%)]	Loss: 0.599654
Train Epoch: 8 [50560/60000 (84%)]	Loss: 0.659849
Train Epoch: 8 [51200/60000 (85%)]	Loss: 0.595036
Train Epoch: 8 [51840/60000 (86%)]	Loss: 0.672949
Train Epoch: 8 [52480/60000 (87%)]	Loss: 0.613096
Train Epoch: 8 [53120/60000 (88%)]	Loss: 0.668293
Train Epoch: 8 [53760/60000 (90%)]	Loss: 0.666945
Train Epoch: 8 [54400/60000 (91%)]	Loss: 0.675986
Train Epoch: 8 [55040/60000 (92%)]	Loss: 0.637825
Train Epoch: 8 [55680/60000 (93%)]	Loss: 0.658959
Train Epoch: 8 [56320/60000 (94%)]	Loss: 0.723985
Train Epoch: 8 [56960/60000 (95%)]	Loss: 0.610149
Train Epoch: 8 [57600/60000 (96%)]	Loss: 0.673735
Train Epoch: 8 [58240/60000 (97%)]	Loss: 0.599380
Train Epoch: 8 [58880/60000 (98%)]	Loss: 0.647105
Train Epoch: 8 [59520/60000 (99%)]	Loss: 0.667784

Test set: Average loss: 0.5811, Accuracy: 9077/10000 (91%)

Train Epoch: 9 [0/60000 (0%)]	Loss: 0.578826
Train Epoch: 9 [640/60000 (1%)]	Loss: 0.602211
Train Epoch: 9 [1280/60000 (2%)]	Loss: 0.616693
Train Epoch: 9 [1920/60000 (3%)]	Loss: 0.662457
Train Epoch: 9 [2560/60000 (4%)]	Loss: 0.617069
Train Epoch: 9 [3200/60000 (5%)]	Loss: 0.594185
Train Epoch: 9 [3840/60000 (6%)]	Loss: 0.630359
Train Epoch: 9 [4480/60000 (7%)]	Loss: 0.667100
Train Epoch: 9 [5120/60000 (9%)]	Loss: 0.714939
Train Epoch: 9 [5760/60000 (10%)]	Loss: 0.670145
Train Epoch: 9 [6400/60000 (11%)]	Loss: 0.670754
Train Epoch: 9 [7040/60000 (12%)]	Loss: 0.659898
Train Epoch: 9 [7680/60000 (13%)]	Loss: 0.673772
Train Epoch: 9 [8320/60000 (14%)]	Loss: 0.641839
Train Epoch: 9 [8960/60000 (15%)]	Loss: 0.635953
Train Epoch: 9 [9600/60000 (16%)]	Loss: 0.690796
Train Epoch: 9 [10240/60000 (17%)]	Loss: 0.680862
Train Epoch: 9 [10880/60000 (18%)]	Loss: 0.628643
Train Epoch: 9 [11520/60000 (19%)]	Loss: 0.565112
Train Epoch: 9 [12160/60000 (20%)]	Loss: 0.691576
Train Epoch: 9 [12800/60000 (21%)]	Loss: 0.594780
Train Epoch: 9 [13440/60000 (22%)]	Loss: 0.647343
Train Epoch: 9 [14080/60000 (23%)]	Loss: 0.718045
Train Epoch: 9 [14720/60000 (25%)]	Loss: 0.603214
Train Epoch: 9 [15360/60000 (26%)]	Loss: 0.661207
Train Epoch: 9 [16000/60000 (27%)]	Loss: 0.592458
Train Epoch: 9 [16640/60000 (28%)]	Loss: 0.697465
Train Epoch: 9 [17280/60000 (29%)]	Loss: 0.559869
Train Epoch: 9 [17920/60000 (30%)]	Loss: 0.590281
Train Epoch: 9 [18560/60000 (31%)]	Loss: 0.677245
Train Epoch: 9 [19200/60000 (32%)]	Loss: 0.630284
Train Epoch: 9 [19840/60000 (33%)]	Loss: 0.695470
Train Epoch: 9 [20480/60000 (34%)]	Loss: 0.606925
Train Epoch: 9 [21120/60000 (35%)]	Loss: 0.607014
Train Epoch: 9 [21760/60000 (36%)]	Loss: 0.647807
Train Epoch: 9 [22400/60000 (37%)]	Loss: 0.654175
Train Epoch: 9 [23040/60000 (38%)]	Loss: 0.562264
Train Epoch: 9 [23680/60000 (39%)]	Loss: 0.644068
Train Epoch: 9 [24320/60000 (41%)]	Loss: 0.627980
Train Epoch: 9 [24960/60000 (42%)]	Loss: 0.634248
Train Epoch: 9 [25600/60000 (43%)]	Loss: 0.689664
Train Epoch: 9 [26240/60000 (44%)]	Loss: 0.558073
Train Epoch: 9 [26880/60000 (45%)]	Loss: 0.627711
Train Epoch: 9 [27520/60000 (46%)]	Loss: 0.624638
Train Epoch: 9 [28160/60000 (47%)]	Loss: 0.615384
Train Epoch: 9 [28800/60000 (48%)]	Loss: 0.673498
Train Epoch: 9 [29440/60000 (49%)]	Loss: 0.666730
Train Epoch: 9 [30080/60000 (50%)]	Loss: 0.571193
Train Epoch: 9 [30720/60000 (51%)]	Loss: 0.648832
Train Epoch: 9 [31360/60000 (52%)]	Loss: 0.666707
Train Epoch: 9 [32000/60000 (53%)]	Loss: 0.608168
Train Epoch: 9 [32640/60000 (54%)]	Loss: 0.643824
Train Epoch: 9 [33280/60000 (55%)]	Loss: 0.641100
Train Epoch: 9 [33920/60000 (57%)]	Loss: 0.615721
Train Epoch: 9 [34560/60000 (58%)]	Loss: 0.616252
Train Epoch: 9 [35200/60000 (59%)]	Loss: 0.615365
Train Epoch: 9 [35840/60000 (60%)]	Loss: 0.718359
Train Epoch: 9 [36480/60000 (61%)]	Loss: 0.570526
Train Epoch: 9 [37120/60000 (62%)]	Loss: 0.569316
Train Epoch: 9 [37760/60000 (63%)]	Loss: 0.671386
Train Epoch: 9 [38400/60000 (64%)]	Loss: 0.626009
Train Epoch: 9 [39040/60000 (65%)]	Loss: 0.636372
Train Epoch: 9 [39680/60000 (66%)]	Loss: 0.636225
Train Epoch: 9 [40320/60000 (67%)]	Loss: 0.652777
Train Epoch: 9 [40960/60000 (68%)]	Loss: 0.674826
Train Epoch: 9 [41600/60000 (69%)]	Loss: 0.605451
Train Epoch: 9 [42240/60000 (70%)]	Loss: 0.597690
Train Epoch: 9 [42880/60000 (71%)]	Loss: 0.695206
Train Epoch: 9 [43520/60000 (72%)]	Loss: 0.658179
Train Epoch: 9 [44160/60000 (74%)]	Loss: 0.601438
Train Epoch: 9 [44800/60000 (75%)]	Loss: 0.652601
Train Epoch: 9 [45440/60000 (76%)]	Loss: 0.716575
Train Epoch: 9 [46080/60000 (77%)]	Loss: 0.607501
Train Epoch: 9 [46720/60000 (78%)]	Loss: 0.586993
Train Epoch: 9 [47360/60000 (79%)]	Loss: 0.625817
Train Epoch: 9 [48000/60000 (80%)]	Loss: 0.612096
Train Epoch: 9 [48640/60000 (81%)]	Loss: 0.639393
Train Epoch: 9 [49280/60000 (82%)]	Loss: 0.659835
Train Epoch: 9 [49920/60000 (83%)]	Loss: 0.595335
Train Epoch: 9 [50560/60000 (84%)]	Loss: 0.745152
Train Epoch: 9 [51200/60000 (85%)]	Loss: 0.621349
Train Epoch: 9 [51840/60000 (86%)]	Loss: 0.600750
Train Epoch: 9 [52480/60000 (87%)]	Loss: 0.653365
Train Epoch: 9 [53120/60000 (88%)]	Loss: 0.626674
Train Epoch: 9 [53760/60000 (90%)]	Loss: 0.592338
Train Epoch: 9 [54400/60000 (91%)]	Loss: 0.647630
Train Epoch: 9 [55040/60000 (92%)]	Loss: 0.697188
Train Epoch: 9 [55680/60000 (93%)]	Loss: 0.603149
Train Epoch: 9 [56320/60000 (94%)]	Loss: 0.673006
Train Epoch: 9 [56960/60000 (95%)]	Loss: 0.685558
Train Epoch: 9 [57600/60000 (96%)]	Loss: 0.601262
Train Epoch: 9 [58240/60000 (97%)]	Loss: 0.626123
Train Epoch: 9 [58880/60000 (98%)]	Loss: 0.620120
Train Epoch: 9 [59520/60000 (99%)]	Loss: 0.617703

Test set: Average loss: 0.5868, Accuracy: 8868/10000 (89%)

Train Epoch: 10 [0/60000 (0%)]	Loss: 0.650556
Train Epoch: 10 [640/60000 (1%)]	Loss: 0.629223
Train Epoch: 10 [1280/60000 (2%)]	Loss: 0.590180
Train Epoch: 10 [1920/60000 (3%)]	Loss: 0.660610
Train Epoch: 10 [2560/60000 (4%)]	Loss: 0.613488
Train Epoch: 10 [3200/60000 (5%)]	Loss: 0.671348
Train Epoch: 10 [3840/60000 (6%)]	Loss: 0.673508
Train Epoch: 10 [4480/60000 (7%)]	Loss: 0.605496
Train Epoch: 10 [5120/60000 (9%)]	Loss: 0.707410
Train Epoch: 10 [5760/60000 (10%)]	Loss: 0.701333
Train Epoch: 10 [6400/60000 (11%)]	Loss: 0.553390
Train Epoch: 10 [7040/60000 (12%)]	Loss: 0.631245
Train Epoch: 10 [7680/60000 (13%)]	Loss: 0.607966
Train Epoch: 10 [8320/60000 (14%)]	Loss: 0.677582
Train Epoch: 10 [8960/60000 (15%)]	Loss: 0.675460
Train Epoch: 10 [9600/60000 (16%)]	Loss: 0.644316
Train Epoch: 10 [10240/60000 (17%)]	Loss: 0.727695
Train Epoch: 10 [10880/60000 (18%)]	Loss: 0.588102
Train Epoch: 10 [11520/60000 (19%)]	Loss: 0.658106
Train Epoch: 10 [12160/60000 (20%)]	Loss: 0.657907
Train Epoch: 10 [12800/60000 (21%)]	Loss: 0.569159
Train Epoch: 10 [13440/60000 (22%)]	Loss: 0.612892
Train Epoch: 10 [14080/60000 (23%)]	Loss: 0.638120
Train Epoch: 10 [14720/60000 (25%)]	Loss: 0.646345
Train Epoch: 10 [15360/60000 (26%)]	Loss: 0.657896
Train Epoch: 10 [16000/60000 (27%)]	Loss: 0.560318
Train Epoch: 10 [16640/60000 (28%)]	Loss: 0.613955
Train Epoch: 10 [17280/60000 (29%)]	Loss: 0.663677
Train Epoch: 10 [17920/60000 (30%)]	Loss: 0.622272
Train Epoch: 10 [18560/60000 (31%)]	Loss: 0.606338
Train Epoch: 10 [19200/60000 (32%)]	Loss: 0.594631
Train Epoch: 10 [19840/60000 (33%)]	Loss: 0.670725
Train Epoch: 10 [20480/60000 (34%)]	Loss: 0.695312
Train Epoch: 10 [21120/60000 (35%)]	Loss: 0.630764
Train Epoch: 10 [21760/60000 (36%)]	Loss: 0.641154
Train Epoch: 10 [22400/60000 (37%)]	Loss: 0.677001
Train Epoch: 10 [23040/60000 (38%)]	Loss: 0.642654
Train Epoch: 10 [23680/60000 (39%)]	Loss: 0.710414
Train Epoch: 10 [24320/60000 (41%)]	Loss: 0.636245
Train Epoch: 10 [24960/60000 (42%)]	Loss: 0.701951
Train Epoch: 10 [25600/60000 (43%)]	Loss: 0.592652
Train Epoch: 10 [26240/60000 (44%)]	Loss: 0.644269
Train Epoch: 10 [26880/60000 (45%)]	Loss: 0.617567
Train Epoch: 10 [27520/60000 (46%)]	Loss: 0.577226
Train Epoch: 10 [28160/60000 (47%)]	Loss: 0.622442
Train Epoch: 10 [28800/60000 (48%)]	Loss: 0.652562
Train Epoch: 10 [29440/60000 (49%)]	Loss: 0.778824
Train Epoch: 10 [30080/60000 (50%)]	Loss: 0.626428
Train Epoch: 10 [30720/60000 (51%)]	Loss: 0.625038
Train Epoch: 10 [31360/60000 (52%)]	Loss: 0.701506
Train Epoch: 10 [32000/60000 (53%)]	Loss: 0.630832
Train Epoch: 10 [32640/60000 (54%)]	Loss: 0.579440
Train Epoch: 10 [33280/60000 (55%)]	Loss: 0.653421
Train Epoch: 10 [33920/60000 (57%)]	Loss: 0.562746
Train Epoch: 10 [34560/60000 (58%)]	Loss: 0.533638
Train Epoch: 10 [35200/60000 (59%)]	Loss: 0.580179
Train Epoch: 10 [35840/60000 (60%)]	Loss: 0.663329
Train Epoch: 10 [36480/60000 (61%)]	Loss: 0.653651
Train Epoch: 10 [37120/60000 (62%)]	Loss: 0.557627
Train Epoch: 10 [37760/60000 (63%)]	Loss: 0.682302
Train Epoch: 10 [38400/60000 (64%)]	Loss: 0.569992
Train Epoch: 10 [39040/60000 (65%)]	Loss: 0.603091
Train Epoch: 10 [39680/60000 (66%)]	Loss: 0.598774
Train Epoch: 10 [40320/60000 (67%)]	Loss: 0.629422
Train Epoch: 10 [40960/60000 (68%)]	Loss: 0.567804
Train Epoch: 10 [41600/60000 (69%)]	Loss: 0.575616
Train Epoch: 10 [42240/60000 (70%)]	Loss: 0.633374
Train Epoch: 10 [42880/60000 (71%)]	Loss: 0.653221
Train Epoch: 10 [43520/60000 (72%)]	Loss: 0.637741
Train Epoch: 10 [44160/60000 (74%)]	Loss: 0.633974
Train Epoch: 10 [44800/60000 (75%)]	Loss: 0.601659
Train Epoch: 10 [45440/60000 (76%)]	Loss: 0.630615
Train Epoch: 10 [46080/60000 (77%)]	Loss: 0.623496
Train Epoch: 10 [46720/60000 (78%)]	Loss: 0.673298
Train Epoch: 10 [47360/60000 (79%)]	Loss: 0.592828
Train Epoch: 10 [48000/60000 (80%)]	Loss: 0.612317
Train Epoch: 10 [48640/60000 (81%)]	Loss: 0.687894
Train Epoch: 10 [49280/60000 (82%)]	Loss: 0.614512
Train Epoch: 10 [49920/60000 (83%)]	Loss: 0.625078
Train Epoch: 10 [50560/60000 (84%)]	Loss: 0.670156
Train Epoch: 10 [51200/60000 (85%)]	Loss: 0.610463
Train Epoch: 10 [51840/60000 (86%)]	Loss: 0.647936
Train Epoch: 10 [52480/60000 (87%)]	Loss: 0.670010
Train Epoch: 10 [53120/60000 (88%)]	Loss: 0.592359
Train Epoch: 10 [53760/60000 (90%)]	Loss: 0.680630
Train Epoch: 10 [54400/60000 (91%)]	Loss: 0.655934
Train Epoch: 10 [55040/60000 (92%)]	Loss: 0.611184
Train Epoch: 10 [55680/60000 (93%)]	Loss: 0.641515
Train Epoch: 10 [56320/60000 (94%)]	Loss: 0.615761
Train Epoch: 10 [56960/60000 (95%)]	Loss: 0.616890
Train Epoch: 10 [57600/60000 (96%)]	Loss: 0.600154
Train Epoch: 10 [58240/60000 (97%)]	Loss: 0.655065
Train Epoch: 10 [58880/60000 (98%)]	Loss: 0.613596
Train Epoch: 10 [59520/60000 (99%)]	Loss: 0.688135

Test set: Average loss: 0.5816, Accuracy: 9129/10000 (91%)

Train Epoch: 11 [0/60000 (0%)]	Loss: 0.659530
Train Epoch: 11 [640/60000 (1%)]	Loss: 0.670746
Train Epoch: 11 [1280/60000 (2%)]	Loss: 0.595280
Train Epoch: 11 [1920/60000 (3%)]	Loss: 0.548233
Train Epoch: 11 [2560/60000 (4%)]	Loss: 0.605534
Train Epoch: 11 [3200/60000 (5%)]	Loss: 0.787581
Train Epoch: 11 [3840/60000 (6%)]	Loss: 0.704546
Train Epoch: 11 [4480/60000 (7%)]	Loss: 0.605187
Train Epoch: 11 [5120/60000 (9%)]	Loss: 0.601654
Train Epoch: 11 [5760/60000 (10%)]	Loss: 0.674484
Train Epoch: 11 [6400/60000 (11%)]	Loss: 0.611812
Train Epoch: 11 [7040/60000 (12%)]	Loss: 0.577550
Train Epoch: 11 [7680/60000 (13%)]	Loss: 0.675640
Train Epoch: 11 [8320/60000 (14%)]	Loss: 0.609019
Train Epoch: 11 [8960/60000 (15%)]	Loss: 0.654760
Train Epoch: 11 [9600/60000 (16%)]	Loss: 0.668276
Train Epoch: 11 [10240/60000 (17%)]	Loss: 0.700344
Train Epoch: 11 [10880/60000 (18%)]	Loss: 0.647904
Train Epoch: 11 [11520/60000 (19%)]	Loss: 0.612863
Train Epoch: 11 [12160/60000 (20%)]	Loss: 0.651190
Train Epoch: 11 [12800/60000 (21%)]	Loss: 0.623442
Train Epoch: 11 [13440/60000 (22%)]	Loss: 0.612316
Train Epoch: 11 [14080/60000 (23%)]	Loss: 0.673344
Train Epoch: 11 [14720/60000 (25%)]	Loss: 0.662172
Train Epoch: 11 [15360/60000 (26%)]	Loss: 0.564439
Train Epoch: 11 [16000/60000 (27%)]	Loss: 0.648501
Train Epoch: 11 [16640/60000 (28%)]	Loss: 0.732563
Train Epoch: 11 [17280/60000 (29%)]	Loss: 0.700485
Train Epoch: 11 [17920/60000 (30%)]	Loss: 0.549757
Train Epoch: 11 [18560/60000 (31%)]	Loss: 0.634296
Train Epoch: 11 [19200/60000 (32%)]	Loss: 0.724399
Train Epoch: 11 [19840/60000 (33%)]	Loss: 0.696369
Train Epoch: 11 [20480/60000 (34%)]	Loss: 0.673034
Train Epoch: 11 [21120/60000 (35%)]	Loss: 0.619952
Train Epoch: 11 [21760/60000 (36%)]	Loss: 0.653255
Train Epoch: 11 [22400/60000 (37%)]	Loss: 0.671367
Train Epoch: 11 [23040/60000 (38%)]	Loss: 0.592136
Train Epoch: 11 [23680/60000 (39%)]	Loss: 0.595773
Train Epoch: 11 [24320/60000 (41%)]	Loss: 0.646400
Train Epoch: 11 [24960/60000 (42%)]	Loss: 0.577514
Train Epoch: 11 [25600/60000 (43%)]	Loss: 0.600458
Train Epoch: 11 [26240/60000 (44%)]	Loss: 0.636539
Train Epoch: 11 [26880/60000 (45%)]	Loss: 0.598275
Train Epoch: 11 [27520/60000 (46%)]	Loss: 0.631338
Train Epoch: 11 [28160/60000 (47%)]	Loss: 0.628492
Train Epoch: 11 [28800/60000 (48%)]	Loss: 0.601715
Train Epoch: 11 [29440/60000 (49%)]	Loss: 0.685084
Train Epoch: 11 [30080/60000 (50%)]	Loss: 0.536947
Train Epoch: 11 [30720/60000 (51%)]	Loss: 0.605603
Train Epoch: 11 [31360/60000 (52%)]	Loss: 0.614050
Train Epoch: 11 [32000/60000 (53%)]	Loss: 0.636084
Train Epoch: 11 [32640/60000 (54%)]	Loss: 0.668198
Train Epoch: 11 [33280/60000 (55%)]	Loss: 0.583040
Train Epoch: 11 [33920/60000 (57%)]	Loss: 0.551880
Train Epoch: 11 [34560/60000 (58%)]	Loss: 0.647910
Train Epoch: 11 [35200/60000 (59%)]	Loss: 0.582428
Train Epoch: 11 [35840/60000 (60%)]	Loss: 0.639843
Train Epoch: 11 [36480/60000 (61%)]	Loss: 0.618734
Train Epoch: 11 [37120/60000 (62%)]	Loss: 0.618055
Train Epoch: 11 [37760/60000 (63%)]	Loss: 0.651700
Train Epoch: 11 [38400/60000 (64%)]	Loss: 0.651255
Train Epoch: 11 [39040/60000 (65%)]	Loss: 0.588944
Train Epoch: 11 [39680/60000 (66%)]	Loss: 0.568360
Train Epoch: 11 [40320/60000 (67%)]	Loss: 0.642925
Train Epoch: 11 [40960/60000 (68%)]	Loss: 0.642339
Train Epoch: 11 [41600/60000 (69%)]	Loss: 0.632191
Train Epoch: 11 [42240/60000 (70%)]	Loss: 0.696143
Train Epoch: 11 [42880/60000 (71%)]	Loss: 0.595977
Train Epoch: 11 [43520/60000 (72%)]	Loss: 0.649418
Train Epoch: 11 [44160/60000 (74%)]	Loss: 0.652544
Train Epoch: 11 [44800/60000 (75%)]	Loss: 0.592935
Train Epoch: 11 [45440/60000 (76%)]	Loss: 0.639931
Train Epoch: 11 [46080/60000 (77%)]	Loss: 0.593051
Train Epoch: 11 [46720/60000 (78%)]	Loss: 0.612821
Train Epoch: 11 [47360/60000 (79%)]	Loss: 0.763203
Train Epoch: 11 [48000/60000 (80%)]	Loss: 0.549479
Train Epoch: 11 [48640/60000 (81%)]	Loss: 0.711056
Train Epoch: 11 [49280/60000 (82%)]	Loss: 0.649383
Train Epoch: 11 [49920/60000 (83%)]	Loss: 0.599281
Train Epoch: 11 [50560/60000 (84%)]	Loss: 0.634407
Train Epoch: 11 [51200/60000 (85%)]	Loss: 0.671545
Train Epoch: 11 [51840/60000 (86%)]	Loss: 0.592102
Train Epoch: 11 [52480/60000 (87%)]	Loss: 0.658541
Train Epoch: 11 [53120/60000 (88%)]	Loss: 0.624999
Train Epoch: 11 [53760/60000 (90%)]	Loss: 0.659089
Train Epoch: 11 [54400/60000 (91%)]	Loss: 0.589034
Train Epoch: 11 [55040/60000 (92%)]	Loss: 0.603932
Train Epoch: 11 [55680/60000 (93%)]	Loss: 0.580338
Train Epoch: 11 [56320/60000 (94%)]	Loss: 0.579921
Train Epoch: 11 [56960/60000 (95%)]	Loss: 0.549601
Train Epoch: 11 [57600/60000 (96%)]	Loss: 0.590883
Train Epoch: 11 [58240/60000 (97%)]	Loss: 0.601804
Train Epoch: 11 [58880/60000 (98%)]	Loss: 0.546295
Train Epoch: 11 [59520/60000 (99%)]	Loss: 0.589213

Test set: Average loss: 0.5872, Accuracy: 9023/10000 (90%)

Train Epoch: 12 [0/60000 (0%)]	Loss: 0.595379
Train Epoch: 12 [640/60000 (1%)]	Loss: 0.610888
Train Epoch: 12 [1280/60000 (2%)]	Loss: 0.683008
Train Epoch: 12 [1920/60000 (3%)]	Loss: 0.676074
Train Epoch: 12 [2560/60000 (4%)]	Loss: 0.596891
Train Epoch: 12 [3200/60000 (5%)]	Loss: 0.604284
Train Epoch: 12 [3840/60000 (6%)]	Loss: 0.602326
Train Epoch: 12 [4480/60000 (7%)]	Loss: 0.653764
Train Epoch: 12 [5120/60000 (9%)]	Loss: 0.632554
Train Epoch: 12 [5760/60000 (10%)]	Loss: 0.730525
Train Epoch: 12 [6400/60000 (11%)]	Loss: 0.609275
Train Epoch: 12 [7040/60000 (12%)]	Loss: 0.592552
Train Epoch: 12 [7680/60000 (13%)]	Loss: 0.647419
Train Epoch: 12 [8320/60000 (14%)]	Loss: 0.576635
Train Epoch: 12 [8960/60000 (15%)]	Loss: 0.552821
Train Epoch: 12 [9600/60000 (16%)]	Loss: 0.679484
Train Epoch: 12 [10240/60000 (17%)]	Loss: 0.671567
Train Epoch: 12 [10880/60000 (18%)]	Loss: 0.696895
Train Epoch: 12 [11520/60000 (19%)]	Loss: 0.613405
Train Epoch: 12 [12160/60000 (20%)]	Loss: 0.586853
Train Epoch: 12 [12800/60000 (21%)]	Loss: 0.633846
Train Epoch: 12 [13440/60000 (22%)]	Loss: 0.595330
Train Epoch: 12 [14080/60000 (23%)]	Loss: 0.636249
Train Epoch: 12 [14720/60000 (25%)]	Loss: 0.634204
Train Epoch: 12 [15360/60000 (26%)]	Loss: 0.665415
Train Epoch: 12 [16000/60000 (27%)]	Loss: 0.612937
Train Epoch: 12 [16640/60000 (28%)]	Loss: 0.580748
Train Epoch: 12 [17280/60000 (29%)]	Loss: 0.636935
Train Epoch: 12 [17920/60000 (30%)]	Loss: 0.561105
Train Epoch: 12 [18560/60000 (31%)]	Loss: 0.611820
Train Epoch: 12 [19200/60000 (32%)]	Loss: 0.618383
Train Epoch: 12 [19840/60000 (33%)]	Loss: 0.680626
Train Epoch: 12 [20480/60000 (34%)]	Loss: 0.644620
Train Epoch: 12 [21120/60000 (35%)]	Loss: 0.628237
Train Epoch: 12 [21760/60000 (36%)]	Loss: 0.599269
Train Epoch: 12 [22400/60000 (37%)]	Loss: 0.616120
Train Epoch: 12 [23040/60000 (38%)]	Loss: 0.630054
Train Epoch: 12 [23680/60000 (39%)]	Loss: 0.592902
Train Epoch: 12 [24320/60000 (41%)]	Loss: 0.642904
Train Epoch: 12 [24960/60000 (42%)]	Loss: 0.613132
Train Epoch: 12 [25600/60000 (43%)]	Loss: 0.569488
Train Epoch: 12 [26240/60000 (44%)]	Loss: 0.593515
Train Epoch: 12 [26880/60000 (45%)]	Loss: 0.665423
Train Epoch: 12 [27520/60000 (46%)]	Loss: 0.652273
Train Epoch: 12 [28160/60000 (47%)]	Loss: 0.583993
Train Epoch: 12 [28800/60000 (48%)]	Loss: 0.650644
Train Epoch: 12 [29440/60000 (49%)]	Loss: 0.602503
Train Epoch: 12 [30080/60000 (50%)]	Loss: 0.658704
Train Epoch: 12 [30720/60000 (51%)]	Loss: 0.605581
Train Epoch: 12 [31360/60000 (52%)]	Loss: 0.548887
Train Epoch: 12 [32000/60000 (53%)]	Loss: 0.629813
Train Epoch: 12 [32640/60000 (54%)]	Loss: 0.664389
Train Epoch: 12 [33280/60000 (55%)]	Loss: 0.631786
Train Epoch: 12 [33920/60000 (57%)]	Loss: 0.677031
Train Epoch: 12 [34560/60000 (58%)]	Loss: 0.641032
Train Epoch: 12 [35200/60000 (59%)]	Loss: 0.629934
Train Epoch: 12 [35840/60000 (60%)]	Loss: 0.570495
Train Epoch: 12 [36480/60000 (61%)]	Loss: 0.588242
Train Epoch: 12 [37120/60000 (62%)]	Loss: 0.657254
Train Epoch: 12 [37760/60000 (63%)]	Loss: 0.643323
Train Epoch: 12 [38400/60000 (64%)]	Loss: 0.660849
Train Epoch: 12 [39040/60000 (65%)]	Loss: 0.563105
Train Epoch: 12 [39680/60000 (66%)]	Loss: 0.609043
Train Epoch: 12 [40320/60000 (67%)]	Loss: 0.582163
Train Epoch: 12 [40960/60000 (68%)]	Loss: 0.648290
Train Epoch: 12 [41600/60000 (69%)]	Loss: 0.603814
Train Epoch: 12 [42240/60000 (70%)]	Loss: 0.650669
Train Epoch: 12 [42880/60000 (71%)]	Loss: 0.626788
Train Epoch: 12 [43520/60000 (72%)]	Loss: 0.653780
Train Epoch: 12 [44160/60000 (74%)]	Loss: 0.574759
Train Epoch: 12 [44800/60000 (75%)]	Loss: 0.602522
Train Epoch: 12 [45440/60000 (76%)]	Loss: 0.647370
Train Epoch: 12 [46080/60000 (77%)]	Loss: 0.609551
Train Epoch: 12 [46720/60000 (78%)]	Loss: 0.586001
Train Epoch: 12 [47360/60000 (79%)]	Loss: 0.619178
Train Epoch: 12 [48000/60000 (80%)]	Loss: 0.623998
Train Epoch: 12 [48640/60000 (81%)]	Loss: 0.660458
Train Epoch: 12 [49280/60000 (82%)]	Loss: 0.618719
Train Epoch: 12 [49920/60000 (83%)]	Loss: 0.538114
Train Epoch: 12 [50560/60000 (84%)]	Loss: 0.668856
Train Epoch: 12 [51200/60000 (85%)]	Loss: 0.608685
Train Epoch: 12 [51840/60000 (86%)]	Loss: 0.633522
Train Epoch: 12 [52480/60000 (87%)]	Loss: 0.645036
Train Epoch: 12 [53120/60000 (88%)]	Loss: 0.659189
Train Epoch: 12 [53760/60000 (90%)]	Loss: 0.608432
Train Epoch: 12 [54400/60000 (91%)]	Loss: 0.609206
Train Epoch: 12 [55040/60000 (92%)]	Loss: 0.653395
Train Epoch: 12 [55680/60000 (93%)]	Loss: 0.678427
Train Epoch: 12 [56320/60000 (94%)]	Loss: 0.685559
Train Epoch: 12 [56960/60000 (95%)]	Loss: 0.574823
Train Epoch: 12 [57600/60000 (96%)]	Loss: 0.626118
Train Epoch: 12 [58240/60000 (97%)]	Loss: 0.628645
Train Epoch: 12 [58880/60000 (98%)]	Loss: 0.663587
Train Epoch: 12 [59520/60000 (99%)]	Loss: 0.597958

Test set: Average loss: 0.5827, Accuracy: 9163/10000 (92%)

Train Epoch: 13 [0/60000 (0%)]	Loss: 0.679710
Train Epoch: 13 [640/60000 (1%)]	Loss: 0.624228
Train Epoch: 13 [1280/60000 (2%)]	Loss: 0.612203
Train Epoch: 13 [1920/60000 (3%)]	Loss: 0.655052
Train Epoch: 13 [2560/60000 (4%)]	Loss: 0.614261
Train Epoch: 13 [3200/60000 (5%)]	Loss: 0.606739
Train Epoch: 13 [3840/60000 (6%)]	Loss: 0.585996
Train Epoch: 13 [4480/60000 (7%)]	Loss: 0.640518
Train Epoch: 13 [5120/60000 (9%)]	Loss: 0.636912
Train Epoch: 13 [5760/60000 (10%)]	Loss: 0.621795
Train Epoch: 13 [6400/60000 (11%)]	Loss: 0.619459
Train Epoch: 13 [7040/60000 (12%)]	Loss: 0.714922
Train Epoch: 13 [7680/60000 (13%)]	Loss: 0.682550
Train Epoch: 13 [8320/60000 (14%)]	Loss: 0.633452
Train Epoch: 13 [8960/60000 (15%)]	Loss: 0.645593
Train Epoch: 13 [9600/60000 (16%)]	Loss: 0.609870
Train Epoch: 13 [10240/60000 (17%)]	Loss: 0.573921
Train Epoch: 13 [10880/60000 (18%)]	Loss: 0.666992
Train Epoch: 13 [11520/60000 (19%)]	Loss: 0.593731
Train Epoch: 13 [12160/60000 (20%)]	Loss: 0.625469
Train Epoch: 13 [12800/60000 (21%)]	Loss: 0.545679
Train Epoch: 13 [13440/60000 (22%)]	Loss: 0.689585
Train Epoch: 13 [14080/60000 (23%)]	Loss: 0.627622
Train Epoch: 13 [14720/60000 (25%)]	Loss: 0.599991
Train Epoch: 13 [15360/60000 (26%)]	Loss: 0.596753
Train Epoch: 13 [16000/60000 (27%)]	Loss: 0.674547
Train Epoch: 13 [16640/60000 (28%)]	Loss: 0.689053
Train Epoch: 13 [17280/60000 (29%)]	Loss: 0.628484
Train Epoch: 13 [17920/60000 (30%)]	Loss: 0.630628
Train Epoch: 13 [18560/60000 (31%)]	Loss: 0.590002
Train Epoch: 13 [19200/60000 (32%)]	Loss: 0.603102
Train Epoch: 13 [19840/60000 (33%)]	Loss: 0.642268
Train Epoch: 13 [20480/60000 (34%)]	Loss: 0.562049
Train Epoch: 13 [21120/60000 (35%)]	Loss: 0.683796
Train Epoch: 13 [21760/60000 (36%)]	Loss: 0.622006
Train Epoch: 13 [22400/60000 (37%)]	Loss: 0.651033
Train Epoch: 13 [23040/60000 (38%)]	Loss: 0.669977
Train Epoch: 13 [23680/60000 (39%)]	Loss: 0.590680
Train Epoch: 13 [24320/60000 (41%)]	Loss: 0.619086
Train Epoch: 13 [24960/60000 (42%)]	Loss: 0.622803
Train Epoch: 13 [25600/60000 (43%)]	Loss: 0.611591
Train Epoch: 13 [26240/60000 (44%)]	Loss: 0.619045
Train Epoch: 13 [26880/60000 (45%)]	Loss: 0.610335
Train Epoch: 13 [27520/60000 (46%)]	Loss: 0.657733
Train Epoch: 13 [28160/60000 (47%)]	Loss: 0.655460
Train Epoch: 13 [28800/60000 (48%)]	Loss: 0.689685
Train Epoch: 13 [29440/60000 (49%)]	Loss: 0.641916
Train Epoch: 13 [30080/60000 (50%)]	Loss: 0.569771
Train Epoch: 13 [30720/60000 (51%)]	Loss: 0.615312
Train Epoch: 13 [31360/60000 (52%)]	Loss: 0.597685
Train Epoch: 13 [32000/60000 (53%)]	Loss: 0.615204
Train Epoch: 13 [32640/60000 (54%)]	Loss: 0.588518
Train Epoch: 13 [33280/60000 (55%)]	Loss: 0.638485
Train Epoch: 13 [33920/60000 (57%)]	Loss: 0.658427
Train Epoch: 13 [34560/60000 (58%)]	Loss: 0.604269
Train Epoch: 13 [35200/60000 (59%)]	Loss: 0.616714
Train Epoch: 13 [35840/60000 (60%)]	Loss: 0.655457
Train Epoch: 13 [36480/60000 (61%)]	Loss: 0.658981
Train Epoch: 13 [37120/60000 (62%)]	Loss: 0.635588
Train Epoch: 13 [37760/60000 (63%)]	Loss: 0.663484
Train Epoch: 13 [38400/60000 (64%)]	Loss: 0.621714
Train Epoch: 13 [39040/60000 (65%)]	Loss: 0.604448
Train Epoch: 13 [39680/60000 (66%)]	Loss: 0.575177
Train Epoch: 13 [40320/60000 (67%)]	Loss: 0.674552
Train Epoch: 13 [40960/60000 (68%)]	Loss: 0.635782
Train Epoch: 13 [41600/60000 (69%)]	Loss: 0.626080
Train Epoch: 13 [42240/60000 (70%)]	Loss: 0.646283
Train Epoch: 13 [42880/60000 (71%)]	Loss: 0.745756
Train Epoch: 13 [43520/60000 (72%)]	Loss: 0.636601
Train Epoch: 13 [44160/60000 (74%)]	Loss: 0.678499
Train Epoch: 13 [44800/60000 (75%)]	Loss: 0.565497
Train Epoch: 13 [45440/60000 (76%)]	Loss: 0.578808
Train Epoch: 13 [46080/60000 (77%)]	Loss: 0.565150
Train Epoch: 13 [46720/60000 (78%)]	Loss: 0.595105
Train Epoch: 13 [47360/60000 (79%)]	Loss: 0.705441
Train Epoch: 13 [48000/60000 (80%)]	Loss: 0.786994
Train Epoch: 13 [48640/60000 (81%)]	Loss: 0.716574
Train Epoch: 13 [49280/60000 (82%)]	Loss: 0.643951
Train Epoch: 13 [49920/60000 (83%)]	Loss: 0.696411
Train Epoch: 13 [50560/60000 (84%)]	Loss: 0.606691
Train Epoch: 13 [51200/60000 (85%)]	Loss: 0.645686
Train Epoch: 13 [51840/60000 (86%)]	Loss: 0.704005
Train Epoch: 13 [52480/60000 (87%)]	Loss: 0.573024
Train Epoch: 13 [53120/60000 (88%)]	Loss: 0.642730
Train Epoch: 13 [53760/60000 (90%)]	Loss: 0.611377
Train Epoch: 13 [54400/60000 (91%)]	Loss: 0.562085
Train Epoch: 13 [55040/60000 (92%)]	Loss: 0.589696
Train Epoch: 13 [55680/60000 (93%)]	Loss: 0.600371
Train Epoch: 13 [56320/60000 (94%)]	Loss: 0.618404
Train Epoch: 13 [56960/60000 (95%)]	Loss: 0.584765
Train Epoch: 13 [57600/60000 (96%)]	Loss: 0.645296
Train Epoch: 13 [58240/60000 (97%)]	Loss: 0.574155
Train Epoch: 13 [58880/60000 (98%)]	Loss: 0.637868
Train Epoch: 13 [59520/60000 (99%)]	Loss: 0.596308

Test set: Average loss: 0.5835, Accuracy: 9155/10000 (92%)

Train Epoch: 14 [0/60000 (0%)]	Loss: 0.585952
Train Epoch: 14 [640/60000 (1%)]	Loss: 0.701281
Train Epoch: 14 [1280/60000 (2%)]	Loss: 0.658894
Train Epoch: 14 [1920/60000 (3%)]	Loss: 0.662308
Train Epoch: 14 [2560/60000 (4%)]	Loss: 0.633019
Train Epoch: 14 [3200/60000 (5%)]	Loss: 0.711046
Train Epoch: 14 [3840/60000 (6%)]	Loss: 0.657272
Train Epoch: 14 [4480/60000 (7%)]	Loss: 0.656501
Train Epoch: 14 [5120/60000 (9%)]	Loss: 0.635089
Train Epoch: 14 [5760/60000 (10%)]	Loss: 0.785942
Train Epoch: 14 [6400/60000 (11%)]	Loss: 0.592267
Train Epoch: 14 [7040/60000 (12%)]	Loss: 0.668679
Train Epoch: 14 [7680/60000 (13%)]	Loss: 0.618138
Train Epoch: 14 [8320/60000 (14%)]	Loss: 0.609593
Train Epoch: 14 [8960/60000 (15%)]	Loss: 0.612504
Train Epoch: 14 [9600/60000 (16%)]	Loss: 0.624620
Train Epoch: 14 [10240/60000 (17%)]	Loss: 0.675233
Train Epoch: 14 [10880/60000 (18%)]	Loss: 0.591791
Train Epoch: 14 [11520/60000 (19%)]	Loss: 0.585406
Train Epoch: 14 [12160/60000 (20%)]	Loss: 0.587335
Train Epoch: 14 [12800/60000 (21%)]	Loss: 0.583577
Train Epoch: 14 [13440/60000 (22%)]	Loss: 0.624970
Train Epoch: 14 [14080/60000 (23%)]	Loss: 0.644836
Train Epoch: 14 [14720/60000 (25%)]	Loss: 0.608672
Train Epoch: 14 [15360/60000 (26%)]	Loss: 0.584978
Train Epoch: 14 [16000/60000 (27%)]	Loss: 0.576684
Train Epoch: 14 [16640/60000 (28%)]	Loss: 0.607673
Train Epoch: 14 [17280/60000 (29%)]	Loss: 0.547107
Train Epoch: 14 [17920/60000 (30%)]	Loss: 0.672443
Train Epoch: 14 [18560/60000 (31%)]	Loss: 0.564704
Train Epoch: 14 [19200/60000 (32%)]	Loss: 0.645698
Train Epoch: 14 [19840/60000 (33%)]	Loss: 0.612638
Train Epoch: 14 [20480/60000 (34%)]	Loss: 0.653497
Train Epoch: 14 [21120/60000 (35%)]	Loss: 0.660650
Train Epoch: 14 [21760/60000 (36%)]	Loss: 0.585572
Train Epoch: 14 [22400/60000 (37%)]	Loss: 0.560714
Train Epoch: 14 [23040/60000 (38%)]	Loss: 0.583192
Train Epoch: 14 [23680/60000 (39%)]	Loss: 0.599548
Train Epoch: 14 [24320/60000 (41%)]	Loss: 0.580376
Train Epoch: 14 [24960/60000 (42%)]	Loss: 0.626721
Train Epoch: 14 [25600/60000 (43%)]	Loss: 0.641809
Train Epoch: 14 [26240/60000 (44%)]	Loss: 0.597601
Train Epoch: 14 [26880/60000 (45%)]	Loss: 0.635401
Train Epoch: 14 [27520/60000 (46%)]	Loss: 0.669811
Train Epoch: 14 [28160/60000 (47%)]	Loss: 0.616458
Train Epoch: 14 [28800/60000 (48%)]	Loss: 0.665900
Train Epoch: 14 [29440/60000 (49%)]	Loss: 0.611768
Train Epoch: 14 [30080/60000 (50%)]	Loss: 0.603277
Train Epoch: 14 [30720/60000 (51%)]	Loss: 0.628918
Train Epoch: 14 [31360/60000 (52%)]	Loss: 0.616099
Train Epoch: 14 [32000/60000 (53%)]	Loss: 0.705863
Train Epoch: 14 [32640/60000 (54%)]	Loss: 0.590716
Train Epoch: 14 [33280/60000 (55%)]	Loss: 0.621051
Train Epoch: 14 [33920/60000 (57%)]	Loss: 0.518643
Train Epoch: 14 [34560/60000 (58%)]	Loss: 0.601564
Train Epoch: 14 [35200/60000 (59%)]	Loss: 0.575993
Train Epoch: 14 [35840/60000 (60%)]	Loss: 0.652142
Train Epoch: 14 [36480/60000 (61%)]	Loss: 0.706018
Train Epoch: 14 [37120/60000 (62%)]	Loss: 0.658774
Train Epoch: 14 [37760/60000 (63%)]	Loss: 0.636433
Train Epoch: 14 [38400/60000 (64%)]	Loss: 0.685398
Train Epoch: 14 [39040/60000 (65%)]	Loss: 0.625651
Train Epoch: 14 [39680/60000 (66%)]	Loss: 0.588738
Train Epoch: 14 [40320/60000 (67%)]	Loss: 0.594724
Train Epoch: 14 [40960/60000 (68%)]	Loss: 0.643608
Train Epoch: 14 [41600/60000 (69%)]	Loss: 0.648216
Train Epoch: 14 [42240/60000 (70%)]	Loss: 0.613402
Train Epoch: 14 [42880/60000 (71%)]	Loss: 0.580880
Train Epoch: 14 [43520/60000 (72%)]	Loss: 0.610102
Train Epoch: 14 [44160/60000 (74%)]	Loss: 0.602010
Train Epoch: 14 [44800/60000 (75%)]	Loss: 0.612606
Train Epoch: 14 [45440/60000 (76%)]	Loss: 0.660709
Train Epoch: 14 [46080/60000 (77%)]	Loss: 0.571197
Train Epoch: 14 [46720/60000 (78%)]	Loss: 0.630411
Train Epoch: 14 [47360/60000 (79%)]	Loss: 0.843941
Train Epoch: 14 [48000/60000 (80%)]	Loss: 0.558983
Train Epoch: 14 [48640/60000 (81%)]	Loss: 0.609951
Train Epoch: 14 [49280/60000 (82%)]	Loss: 0.669133
Train Epoch: 14 [49920/60000 (83%)]	Loss: 0.567263
Train Epoch: 14 [50560/60000 (84%)]	Loss: 0.672197
Train Epoch: 14 [51200/60000 (85%)]	Loss: 0.720792
Train Epoch: 14 [51840/60000 (86%)]	Loss: 0.619728
Train Epoch: 14 [52480/60000 (87%)]	Loss: 0.701936
Train Epoch: 14 [53120/60000 (88%)]	Loss: 0.677871
Train Epoch: 14 [53760/60000 (90%)]	Loss: 0.621737
Train Epoch: 14 [54400/60000 (91%)]	Loss: 0.584803
Train Epoch: 14 [55040/60000 (92%)]	Loss: 0.633493
Train Epoch: 14 [55680/60000 (93%)]	Loss: 0.604813
Train Epoch: 14 [56320/60000 (94%)]	Loss: 0.660941
Train Epoch: 14 [56960/60000 (95%)]	Loss: 0.661254
Train Epoch: 14 [57600/60000 (96%)]	Loss: 0.659504
Train Epoch: 14 [58240/60000 (97%)]	Loss: 0.629760
Train Epoch: 14 [58880/60000 (98%)]	Loss: 0.661404
Train Epoch: 14 [59520/60000 (99%)]	Loss: 0.678125

Test set: Average loss: 0.5825, Accuracy: 9168/10000 (92%)