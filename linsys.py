import numpy as np
import matplotlib.pyplot as plt

from scipy.sparse import dok_matrix
from scipy.sparse.linalg import spsolve
from scipy.sparse.linalg import factorized

from nodenet import Net, build_boundary, build_interior, plot_net, Nodetype, label_nodes

def gen_U_h(net, u0=lambda x,y: 0):

    N = net.k
    U = np.zeros(N, dtype=float)

    for p in net.nodes:
        U[p.k] = u0(p.x, p.y)

    return U

def gen_F_h(net, f, gd=lambda x,y: 0, gn=lambda x,y: 0):

    N = net.k
    F = np.zeros(N, dtype=float)

    for p in net.nodes:
        if p.nodetype == Nodetype.DIRICH: # Dirichlet boundary conditions
            F[p.k] = gd(p.x, p.y)

        elif p.nodetype == Nodetype.NEUMANN: # Neumann boundary conditions
            F[p.k] = gn(p.x, p.y)
            
        elif p.nodetype == Nodetype.INNER:
            F[p.k] = net.h**2 * f(p.x, p.y)
            
        elif p.nodetype == Nodetype.EDGE:
            F[p.k] = net.h**2 * f(p.x, p.y)
            
        else:
            raise RuntimeError('missing nodetype')
    
    return F

def gen_A_h(net):

    N = net.k
    N2 = N**2

    A = dok_matrix((N,N), dtype=float)

    for P in net.nodes:
        p = P.k

        if P.nodetype == Nodetype.INNER:
            n, e, s, w = P.N.k, P.E.k, P.S.k, P.W.k
            A[p,n] = 1.0
            A[p,e] = 1.0
            A[p,s] = 1.0
            A[p,w] = 1.0
            A[p,p] = -4.0


        elif P.nodetype == Nodetype.EDGE:
            n, e, s, w = P.N.k, P.E.k, P.S.k, P.W.k
            xi_n = ( P.N.y - P.y ) / net.h
            xi_e = ( P.E.x - P.x ) / net.h
            xi_n_inv = 1.0 / xi_n
            xi_e_inv = 1.0 / xi_e
            chi_n = 2.0 / (1.0 + xi_n)
            chi_e = 2.0 / (1.0 + xi_e)

            A[p,n] = chi_n * xi_n_inv
            A[p,e] = chi_e * xi_e_inv
            A[p,s] = chi_n
            A[p,w] = chi_e
            A[p,p] = -chi_n * (1.0 + xi_n_inv) - chi_e * (1.0 + xi_e_inv)


        elif P.nodetype == Nodetype.DIRICH:
            A[p,p] = 1.0


        elif P.nodetype == Nodetype.NEUMANN:
            theta = np.arctan( 1 / (2*P.x) )
            if P.W != None:
                dx = P.x - P.W.x
                dy = net.h
            else:
                dx = net.h
                dy = P.y - P.S.y

            if P.S != None:
                R = P.S.W
            else:
                R = P.W.S
            r = R.k

            if dx / np.cos(theta) < dy / np.sin(theta): # x-intersection is closest
                # Approximate Q by R and R.N
                rn = R.N.k
                d = dx / np.cos(theta)
                hp = d * np.sin(theta)

                A[p,r]  = -hp / (net.h * d)
                A[p,rn] = (hp - net.h) / (net.h * d)
                A[p,p]  = 1 / d

            
            else: # y-intersection closest
                # Approximate Q by R and R.E
                re = R.E.k
                d = dy / np.sin(theta)
                hp = d*np.cos(theta)

                A[p,r]  = -hp / (net.h * d)
                A[p,re] = (hp - net.h) / (net.h * d)
                A[p,p]  = 1 / d



        else:
            raise RuntimeError('missing nodetype')

    return A



def main():
    
    M = 2

    net = Net(M)

    build_interior(net)
    build_boundary(net)
    label_nodes(net)

    plot_net(net)

    plt.show()

    f = lambda x,y: x**2 + y - 1
    U_h = gen_U_h(net, u0=f)
    print(U_h)
    F_h = gen_F_h(net, f=f)
    print(F_h)
    A_h = gen_A_h(net)
    print(A_h.todense())

    plt.show()
    
    return

if __name__ == "__main__":
    main()
