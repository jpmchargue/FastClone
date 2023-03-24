import time
from 

from dataio import SpeakerDataset, UtteranceDataset


def train_gel(verifier, num_iters, learning_rate=0.01, print_every=5):
    start = time.time()

    #criterion = GE2ELoss(init_w=10.0, init_b=-5.0, loss_method='contrast')
    criterion = nn.L1Loss()
    optimizer_v = optim.SGD(verifier.parameters(), lr=learning_rate)
    loss_records = []
    loss_total = 0
    
    # generate input tensor
    dataset = SpeakerDataset(params.DATA_PATH)

    # run step_gel()
    #input_batch = torch.from_numpy(dataset[0]).to(device)
    for iter in range(1, num_iters + 1):
        print("loading batch")
        input_batch = torch.from_numpy(dataset[iter]).to(device)
        loss = step_gel(input_batch, verifier, optimizer_v) 
        #(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)
        loss_total += loss
        loss_records.append(loss.item())
        print(f"{iter} | LOSS: {loss}")

        if iter % print_every == 0:
            loss_avg = loss_total / print_every
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / num_iters),
                                         iter, iter / num_iters * 100, loss_avg))
            loss_total = 0
    
    plt.plot(loss_records)
    plt.show()