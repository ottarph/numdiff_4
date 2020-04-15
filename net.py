import numpy as np
import matplotlib.pyplot as plt


def netGen(M):

    L = np.linspace(0, 1, M, dtype=float)
    net = np.array([L, np.zeros_like(L)], dtype=float) # y = 0

    # interior points
    X, Y = np.meshgrid(L, L)
    inds = Y < 1 - X**2
    X, Y = X[inds], Y[inds]

    # y = 0
    X = np.append(X, np.copy(L))
    Y = np.append(Y, np.zeros_like(L))

    # x = 0
    X = np.append(X, np.zeros_like(L)) 
    Y = np.append(Y, np.copy(L))

    # x = x_i, y = 1 - x**2
    X = np.append(X, np.copy(L))
    Y = np.append(Y, 1 - L**2)

    # y = y_j, x = sqrt(1 - y)
    X = np.append(X, np.sqrt(1 - L))
    Y = np.append(Y, np.copy(L))


    net = np.array([X, Y]).T

    net = np.unique(net, axis=0)
    

    return net, X, Y

def netGenIndDict(M):

    L = np.linspace(0, 1, M, dtype=float)
    net = np.array([L, np.zeros_like(L)], dtype=float) # y = 0

    inds = dict()

    # interior points
    Y, X = np.meshgrid(L, L)
    print(X)
    Ix, Iy = np.where(Y < 1 - X**2)
    print(Ix, Iy)
    for ix, iy in zip(Ix, Iy):
        print(Y[ix,iy], 1 - X[ix,iy]**2)
    I = Y < 1 - X**2
    X, Y = X[I], Y[I]

    # y = 0
    X = np.append(X, np.copy(L))
    Y = np.append(Y, np.zeros_like(L))

    # x = 0
    X = np.append(X, np.zeros_like(L)) 
    Y = np.append(Y, np.copy(L))

    # x = x_i, y = 1 - x**2
    X = np.append(X, np.copy(L))
    Y = np.append(Y, 1 - L**2)

    # y = y_j, x = sqrt(1 - y)
    X = np.append(X, np.sqrt(1 - L))
    Y = np.append(Y, np.copy(L))


    net = np.array([X, Y]).T

    net = np.unique(net, axis=0)
    

    return net, X, Y, inds

def dictplay(M):
    h = 1/M

    inds = dict()

    k = 0
    xx, yy = [], []
    for i in range(M+1):
        x = i*h
        for j in range(M+1):
            y = j*h

            if y < 1 - x**2:
                inds[(i,j)] = k
                xx.append(x)
                yy.append(y)
                k += 1
            else:
                break


    print(xx)
    print(yy)
    return inds, xx, yy

    


def main():

    M = 5

    '''
    net, X, Y = netGen(M)
    print(net.shape)

    plt.scatter(net[:,0], net[:,1], color='black', s=5**2)
    plt.plot([0,0], [0,1], color='black', linewidth=0.7)
    plt.plot([0,1], [0,0], color='black', linewidth=0.7)
    l = np.linspace(0, 1, 100)
    plt.plot(l, 1 - l**2, color='black', linewidth=0.7)
    plt.grid()
    '''

    #net, X, Y, inds = netGenIndDict(M)
    #print(inds)

    inds, xx, yy = dictplay(M)
    print(inds)
    h = 1/M
    for ij, k in inds.items():
        print(ij)
        print(k)
        x, y = xx[k], yy[k]
        #plt.text(x,y,ij)
        plt.text(x,y,np.round(1 - x**2 - y,3))



    plt.show()


    return

if __name__ == '__main__':
    main()
