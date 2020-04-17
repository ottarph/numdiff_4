import numpy as np
import matplotlib.pyplot as plt

class Node:

    def __init__(self, i, j, x, y, k):
        self.i = i
        self.j = j
        self.x = x
        self.y = y
        self.k = k

        self.N = None
        self.E = None
        self.S = None
        self.W = None
    
        self.neighbours = {'N': self.N, 'E': self.E, 'S': self.S, 'W': self.W}

    def __repr__(self):
        return f'({self.i}, {self.j})'
    
    def add_neighbours(self, N=None, E=None, S=None, W=None):
        if N != None:
            print('ye')
            self.N = N
        if E != None:
            self.E = E
        if S != None:
            self.S = S
        if W != None:
            self.W = W
        self.neighbours = {'N': self.N, 'E': self.E, 'S': self.S, 'W': self.W}


class Net:


    def __init__(self, M):
        self.M = M
        self.k = 0
        self.nodes = []
        self.grid = dict()

    def __repr__(self):
        return str([str(p) for p in self.nodes])

    def add_int_node(self, i, j, x, y):
        p = Node(i, j, x, y, self.k)
        self.nodes.append(p)
        self.grid[(i,j)] = p
        self.k += 1

    def add_edge_nodes(self, q):
        
        if q.N == None:
            x = q.x
            y = 1 - x**2
            n = Node(None, None, x, y, self.k)
            self.nodes.append(n)
            self.k += 1
            q.N = n
            n.S = q
        if q.E == None:
            y = q.y
            x = np.sqrt(1 - y)
            e = Node(None, None, x, y, self.k)
            self.nodes.append(e)
            self.k += 1
            q.E = e
            e.W = q


def build_interior(net):

    M = net.M
    h = 1/M
    for i in range(M):
        x = i*h
        for j in range(M):
            y = j*h
            if y < 1 - x**2:
                net.add_int_node(i,j,x,y)
    
    atlas = net.grid.keys()
    for i in range(M):
        for j in range(M):
            if (i,j) not in atlas:
                break
            p = net.grid[(i,j)]
            n = (i,j+1)
            e = (i+1,j)
            s = (i,j-1)
            w = (i-1,j)
            if n in atlas:
                p.N = net.grid[n]
            if e in atlas:
                p.E = net.grid[e]
            if s in atlas:
                p.S = net.grid[s]
            if w in atlas:
                p.W = net.grid[w]

def build_boundary(net):
    
    M = net.M
    h = 1/M
    atlas = net.grid.keys()

    for i in range(M):
        j = max(filter(lambda ij: ij[0]==i, atlas), key= lambda ij: ij[1])[1]
        q = net.grid[(i,j)]
        net.add_edge_nodes(q)

    for j in range(M):
        i = max(filter(lambda ij: ij[1]==j, atlas), key= lambda ij: ij[0])[0]
        q = net.grid[(i,j)]
        net.add_edge_nodes(q)
    


def main():

    net = Net(10)
    build_interior(net)
    build_boundary(net)

    for p in net.nodes:
        plt.text(p.x, p.y, (p.k))
        plt.scatter(p.x, p.y, s=16)
        for q in [p.N, p.E, p.S, p.W]:
            if q != None:
                plt.plot([p.x,q.x], [p.y, q.y], 'k-', linewidth=0.4)

    xx = np.linspace(0,1,100)
    yy = 1 - xx**2
    plt.plot(xx, yy, 'k--', linewidth=0.6, label='$\partial \Omega$')
    
    plt.legend()
    plt.grid()
    plt.show()


    return


if __name__ == '__main__':
    main()
