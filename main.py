import torch
import torch.nn as nn
import torch.nn.functional as F
from multiprocessing import cpu_count
from torchvision.models import mobilenet_v2
from torchmeta.datasets.helpers import omniglot,cifar_fs
from torchmeta.utils.data import BatchMetaDataLoader
from core import savedic, addvalue
from net.arcface import Arcface
from net.radam import RAdam
from core import addvalue, savedic


def operate():
    for bch, batch in enumerate(dataloader):
        if bch>batch_on_epoch:
            break
        for phase in ['train', 'test']:
            with torch.set_grad_enabled(phase == 'train'):
                data, target = batch[phase]
                # data = data.repeat(1, 1, 3, 1, 1)
                B, ShotWay, C, H, W = data.shape
                assert ShotWay == Shot * Way
                data = data.to(device).reshape(B, Way, Shot, C, H, W).permute(0, 2, 1, 3, 4, 5).reshape(-1, C, H, W)
                target = target.to(device).reshape(B, Way, Shot).permute(0, 2, 1)

                output = model(data)
                output = output.reshape(B, Shot, Way, -1)
                output, target = arcface(output, target)

                loss = lossf(output, target)
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                with torch.no_grad():
                    acc = (output.argmax(-1) == target).sum().float() / len(target)
                print(f'{phase},{e}:{bch}/{min(batch_on_epoch,len(dataloader))}, {loss.item():.4f}, {acc.item() * 100:.2f}%')

                addvalue(writer, f'loss:{phase}', loss.item(), e)
                addvalue(writer, f'acc:{phase}', acc.item(), e)


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    Way = 2
    Shot = 2
    batchsize = 256
    batch_on_epoch=16
    # dataset = omniglot("../data/metric/", ways=Way, shots=Shot, test_shots=Shot, meta_train=True, download=True)
    dataset = cifar_fs("../data/metric/", ways=Way, shots=Shot, test_shots=Shot, meta_train=True, download=True)
    dataloader = BatchMetaDataLoader(dataset, batch_size=batchsize, num_workers=cpu_count())
    lossf = nn.CrossEntropyLoss()
    num_epoch = 100
    writer = {}
    model = mobilenet_v2().to(device)
    arcface = Arcface()
    optimizer = RAdam(model.parameters())

    for e in range(num_epoch):
        operate()
        savedic(writer, 'data')
