import numpy as np
import matplotlib.pyplot as plt


def plot_domain():

    plt.figure()

    x = np.linspace(0,1,100)
    y0 = np.zeros_like(x)
    y1 = 1 - x**2

    plt.plot(x, y0, color='black', linewidth=0.7, label='$y=0$')
    plt.plot(x, y1, color='black', linewidth=0.7, label='$y = 1 - x^2$')
    plt.plot([0,0], [0,1], color='black', linewidth=0.7)

    plt.fill_between(x, y0, y1, alpha=0.5, label=r'$\Omega$')
    plt.grid()
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.legend()

    return

def main():

    plot_domain()
    
    plt.show()

    return


if __name__ == '__main__':
    main()
