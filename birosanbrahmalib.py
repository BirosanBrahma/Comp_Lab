#####   BIROSAN BRAHMA   2311053   #####
#####    MY LIBRARY     #####

import numpy as np
import matplotlib.pyplot as plt 



class mycomplex():
    def __init__(self,real,imag=0.0):
        self.r=real
        self.i=imag
    def display(self):
        print(self.r,", ",self.i,'j',sep='')
    def add(self,c1,c2):
        self.r=c1.r+c2.r
        self.i=c1.i+c2.i
        return(mycomplex(self))
    def sub(self,c1,c2):
        self.r=c1.r-c2.r
        self.i=c1.i-c2.i
        return(mycomplex(self))
    def mul(self,c1,c2):
        self.r=c1.r*c2.r-c1.i*c2.i
        self.i=c1.i*c2.r+c1.r*c2.i
        return(mycomplex(self))
    def mod(self):
        return np.sqrt(self.r**2+self.i**2)

##############################################################

class matrixoperations():
    # ===== Read Matrix from File =====
    def read_from_file(self, filename):
        """Read a matrix from a text file."""
        with open(filename, 'r') as f:
            matrix = []
            for line in f:
                row = [float(num) for num in line.strip().split()]
                matrix.append(row)
        return matrix

    # ===== Matrix Multiplication =====
    def multiply(self, A, B):
        n = len(A)
        m = len(A[0])
        p = len(B[0])

        if len(B) != m:
            raise ValueError("Number of columns in A must equal number of rows in B")
        
        C = [[0 for _ in range(p)] for _ in range(n)]
        for i in range(n):
            for j in range(p):
                for k in range(m):
                    C[i][j] += A[i][k] * B[k][j]
        return C

    # ===== Dot Product (3×1 Vectors) =====
    def dot_product(self, A, B):
        if len(A) != 3 or len(B) != 3:
            raise ValueError("Both vectors must be 3×1 matrices")
        return sum(A[i][0] * B[i][0] for i in range(3))

    # ===== Cross Product (3×1 Vectors) =====
    def cross_product(self, A, B):
        if len(A) != 3 or len(B) != 3:
            raise ValueError("Both vectors must be 3×1 matrices")
        C = [[0], [0], [0]]
        C[0][0] = A[1][0] * B[2][0] - A[2][0] * B[1][0]
        C[1][0] = A[2][0] * B[0][0] - A[0][0] * B[2][0]
        C[2][0] = A[0][0] * B[1][0] - A[1][0] * B[0][0]
        return C

    # ===== Helper: Get Minor =====
    def get_minor(self, matrix, i, j):
        return [row[:j] + row[j+1:] for row in (matrix[:i] + matrix[i+1:])]

    # ===== Determinant =====
    def determinant(self, matrix):
        n = len(matrix)
        if any(len(row) != n for row in matrix):
            raise ValueError("Matrix must be square")
        
        if n == 1:
            return matrix[0][0]
        if n == 2:
            return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
        
        det = 0
        for col in range(n):
            det += ((-1) ** col) * matrix[0][col] * self.determinant(self.get_minor(matrix, 0, col))
        return det

    # ===== Inverse =====
    def inverse(self, matrix):
        n = len(matrix)
        if any(len(row) != n for row in matrix):
            raise ValueError("Matrix must be square")
        
        det = self.determinant(matrix)
        if det == 0:
            raise ValueError("Matrix is singular, cannot find inverse.")
        
        if n == 1:
            return [[1 / matrix[0][0]]]
        
        adj = [[0 for _ in range(n)] for _ in range(n)]
        for i in range(n):
            for j in range(n):
                minor = self.get_minor(matrix, i, j)
                cofactor = ((-1) ** (i + j)) * self.determinant(minor)
                adj[j][i] = cofactor  # Transpose for adjoint
        
        inv = [[adj[i][j] / det for j in range(n)] for i in range(n)]
        return inv



#define x and y , as a python list 
def Plot(x, y , title='Sample Plot', xlabel='X-axis Label', ylabel='Y-axis Label'):
    plt.scatter(x, y, marker='o', color='b')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)  
    plt.show()


def LCG(seed):
    a = 1103515245
    c = 12345
    m = 32768  
    state = int(seed) % m
    L = []
    for _ in range(10000):
        state = (a * state + c) % m
        L.append(state / m)  # normalize only for output

    return L
