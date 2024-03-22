import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt


def main():
    path = r'./test.csv'
    df = pd.read_csv(path, index_col=0)
    target_columns = ['gan_loss', 'gan_ssim', 'gan_psnr']
    cr = df[target_columns].corr()

    top_losses = df['gan_loss'].sort_values()[-25:]
    low_losses = df['gan_loss'].sort_values()[:25]

    ti = np.array([cv2.imread(i) for i in top_losses.index])
    li = np.array([cv2.imread(i) for i in low_losses.index])

    ti_stats = ti.mean(axis=(1, 2)), ti.std(axis=(1, 2))
    li_stats = li.mean(axis=(1, 2)), li.std(axis=(1, 2))

    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))
    colors = ['red', 'green', 'blue']
    ax[0, 0].boxplot(ti_stats[0])
    ax[0, 0].set_xticklabels(colors)
    ax[0, 0].set_title('Channels means (high loss images)')
    ax[0, 0].grid()

    ax[0, 1].boxplot(li_stats[0])
    ax[0, 1].set_xticklabels(colors)
    ax[0, 1].set_title('Channels means (low loss images)')
    ax[0, 1].grid()

    ax[1, 0].boxplot(ti_stats[1])
    ax[1, 0].set_xticklabels(colors)
    ax[1, 0].set_title('Channels variance (high loss images)')
    ax[1, 0].grid()

    ax[1, 1].boxplot(li_stats[1])
    ax[1, 1].set_xticklabels(colors)
    ax[1, 1].set_title('Channels variance (low loss images)')
    ax[1, 1].grid()
    plt.show()



if __name__ == "__main__":
    main()