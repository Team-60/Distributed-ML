# parses training.log
import matplotlib.pyplot as plt

f = open('training.log', 'r')
data = f.read()
f.close()

info = {}
at = 0
while True:
    at = data.find('Process id:', at + 1)
    if at == -1:
        break
    pid = int(data[at + 12: data.find(',', at + 1)])
    at = data.find('Epoch:', at + 1)
    epoch = int(data[at + 7: data.find(' ', at + 8)])
    at = data.find('Loss:', at + 1)
    loss = float(data[at + 6: at + 6 + 8])
    # print(pid, epoch, loss)
    if pid not in info:
        info[pid] = []
    if epoch + 1 > len(info[pid]):
        info[pid].append(loss)
    else:
        info[pid][epoch] = loss

# test metrics
at = data.find('Test set:')
at = data.find('Average Loss', at + 1)
test_loss = data[at + 13: data.find(',', at + 1)]
at = data.find('Accuracy:', at + 1)
test_acc = data[at + 10: data.find('%', at + 1)]

for pid in info:
    plt.plot(range(1, len(info[pid]) + 1), info[pid], label=f'process {pid}')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.title(f'Training Loss vs Epochs; Test loss: {test_loss} & Test acc: {test_acc}')
plt.show()