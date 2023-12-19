from tqdm import tqdm
import time

def test_loader(dataloader, device, num_epochs=30):
    since = time.time()

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        for it,(inputs_, labels_, names, _) in tqdm(enumerate(dataloader)):
            inputs_ = inputs_.to(device)
            labels_ = labels_.to(device)

    time_elapsed = time.time() - since
    print('time_elapsed', time_elapsed)
