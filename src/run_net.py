from training import train_triplet_epoch
from tilenet import make_tilenet

in_channels = 3

model = make_tilenet(in_channels,128)

max_epochs = 10


for epoch in range(max_epochs):
    train_triplet_epoch(model,cuda,dataloader,optimizer,epoch)