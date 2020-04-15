import numpy as np
import matplotlib.pyplot as plt


def plot_domain():

    plt.figure()

    x = np.linspace(0,1,100)
    y0 = np.zeros_like(x)
    y1 = 1 - x**2

    uv = np.array([2*x[::10], np.ones_like(x[::10])])
    uvn = np.linalg.norm(uv, axis=0)
    uv = uv / uvn
    u, v = uv[0], uv[1]

    plt.plot(x, y0, color='black', linewidth=0.7, label='$y=0$')
    plt.plot(x, y1, color='black', linewidth=0.7, label='$y = 1 - x^2$')
    plt.plot([0,0], [0,1], color='black', linewidth=0.7)

    plt.fill_between(x, y0, y1, alpha=0.5, label=r'$\Omega$')

    plt.quiver(x[::10], y1[::10], u, v, angles='xy')

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
