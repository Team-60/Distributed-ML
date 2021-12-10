import sys
from os import listdir
import matplotlib.pyplot as plt

if (len(sys.argv) < 2) or (sys.argv[1] == '--default'):
    dir = 'logs-default'
elif sys.argv[1] == '--tuned':
    dir = 'logs-tuned'
else:
    exit(0)

# collect train losses
losses = {}
for fn in listdir(dir):
    optim = fn[fn.find('-') + 1: fn.rfind('.')] 
    f = open(f'{dir}/{fn}', 'r')
    logs = f.read()
    f.close()

    losses[optim] = []
    at = logs.find('Train Loss:')
    while at != -1:
        num = logs[at + 12: logs.find(';', at) - 1]
        losses[optim].append(float(num))
        at = logs.find('Train Loss:', at + 1)
    
# plot
for optim in losses:
    plt.plot(range(1, len(losses[optim]) + 1), losses[optim], label=optim)
if dir == 'logs-default':
    plt.gca().set_ylim([-0.1, 3.5])
plt.xlabel('epochs')
plt.ylabel('losses')
plt.legend()
plt.title(f'train loss vs epochs [{dir}]')
plt.show()

# collect test losses
losses = {}
for fn in listdir(dir):
    optim = fn[fn.find('-') + 1: fn.rfind('.')] 
    f = open(f'{dir}/{fn}', 'r')
    logs = f.read()
    f.close()

    losses[optim] = []
    at = logs.find('Test Loss:')
    while at != -1:
        num = logs[at + 11: logs.find(';', at) - 1]
        losses[optim].append(float(num))
        at = logs.find('Test Loss:', at + 1)
    
# plot
for optim in losses:
    plt.plot(range(1, len(losses[optim]) + 1), losses[optim], label=optim)
if dir == 'logs-default':
    plt.gca().set_ylim([-0.1, 3.5])
plt.xlabel('epochs')
plt.ylabel('losses')
plt.legend()
plt.title(f'test loss vs epochs [{dir}]')
plt.show()