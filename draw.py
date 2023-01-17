import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

def do_draw(args):
    log_src = args.log_src

    f = pd.read_csv(os.path.join(log_src, 'log.csv'))
    f = np.array(f).tolist()
    epoch = []
    g_loss = []
    d_loss = []

    for each in f:
        epoch.append(each[0])
        g_loss.append(each[1])
        d_loss.append(each[2])
   
    plt.plot(epoch, g_loss, label='Generator Loss')
    plt.plot(epoch, d_loss, label='Discriminator Loss')
    plt.legend(loc='best')
    plt.savefig(os.path.join(log_src, 'result.png'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Drawing')
    parser.add_argument('--log-src', type=str, default='./', help='path of log file')
    args = parser.parse_args()

    do_draw(args)