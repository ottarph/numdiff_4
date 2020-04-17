import numpy as np
import matplotlib.pyplot as plt

from nodenet import Net, build_boundary, build_interior, plot_net, Nodetype, label_nodes

def gen_U_h(net, u0=lambda x,y: 0):

    N = net.k+1
    U = np.zeros(N, dtype=float)

    for p in net.nodes:
        U[p.k] = u0(p.k, p.y)

    return U

def gen_F_h(net, f, gd=lambda x,y: 0, gn=lambda x,y: 0):

    N = net.k+1
    F = np.zeros(N, dtype=float)

    for p in net.nodes:
        if p.nodetype == Nodetype.DIRICH: # Dirichlet boundary conditions
            F[p.k] = gd(p.x, p.y)
            continue
        if p.nodetype == Nodetype.NEUMANN: # Neumann boundary conditions
            raise NotImplementedError
            continue
        if p.nodetype == Nodetype.INNER:
            F[p.k] = net.h**2 * f(p.x, p.y)
            continue
        if p.nodetype == Nodetype.EDGE:
            raise NotImplementedError
            continue
        raise RuntimeError('missing nodetype')
    
    return F


def main():
    
    M = 5

    net = Net(M)
    build_interior(net)
    build_boundary(net)
    plot_net(net)

    label_nodes(net)
    plot_net(net)

    plt.show()

    f = lambda x,y: x**2 + y - 1
    U_h = gen_U_h(net, u0=f)
    print(U_h)
    F_h = gen_F_h(net, f=f)
    print(F_h)

    plt.show()
    
    return

if __name__ == "__main__":
    main()
