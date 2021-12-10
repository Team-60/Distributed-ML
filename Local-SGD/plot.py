# parses training.log
import matplotlib.pyplot as plt

f = open('training.log', 'r')
data = f.read()
f.close()

info = {}
at = 0
while True:
    at = data.find('Rank:', at + 1)
    if at == -1:
        break
    rank = int(data[at + 6: data.find(',', at + 1)])
    at = data.find('Epoch:', at + 1)
    epoch = int(data[at + 7: data.find(' ', at + 8)])
    at = data.find('Loss:', at + 1)
    loss = float(data[at + 6: at + 6 + 8])
    # print(rank, epoch, loss)
    if rank not in info:
        info[rank] = []
    if epoch + 1 > len(info[rank]):
        info[rank].append(loss)
    else:
        info[rank][epoch] = loss

# test metrics
at = data.find('Test set:')
at = data.find('Average Loss', at + 1)
test_loss = data[at + 13: data.find(',', at + 1)]
at = data.find('Accuracy:', at + 1)
test_acc = data[at + 10: data.find('%', at + 1)]

for rank in info:
    plt.plot(range(1, len(info[rank]) + 1), info[rank], label=f'rank {rank}')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.title(f'Training Loss vs Epochs; Test loss: {test_loss} & Test acc: {test_acc}')
plt.show()