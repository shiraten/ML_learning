from typing import Sequence

class Matrix:
    def __init__(self, matrix: Sequence[Sequence[int]]) -> "Matrix":
        # check if all the list values are of int type
        if not all(isinstance(line, Sequence) for line in matrix):
            raise ValueError("Not a matrix. Matrix is following type: Sequence[Sequence[int]]")
        for line in matrix:
            if not all(isinstance(val, int) for val in line):
                raise ValueError("matrix can contain only integer")
        # check all list are same lenght
        len_ref = len(matrix[0])
        if not all(len(line) == len_ref for line in matrix):
            raise ValueError("Not a matrix. matrix must have same number of column in each line")
        self._matrix = matrix
    
    def __repr__(self) -> str:
        return f"Matrix(\n{self._matrix_str(self.matrix)}\n)"
    
    def _matrix_str(self, matrix: Sequence[Sequence[int]]) -> str:
        matrix_str = ""
        for elem in matrix:
            matrix_str += str(elem) + "\n"
        # remove last carriage return
        return matrix_str[:-1]

    @property
    def matrix(self):
        return self._matrix

    @property
    def shape(self):
        lines = len(self._matrix)
        columns = len(self._matrix[0])
        return lines, columns

    @property
    def T(self):
        """
        return Matrix object of the Transpose of the current matrix
        """
        return Matrix([[self._matrix[j][i] for j in range(len(self._matrix))] for i in range(len(self._matrix[0]))])

    def add(self, m: "Matrix"):
        """
        return Matrix object of the addition of the current matrix with another matrix
        """
        if self.shape != m.shape:
            raise AssertionError("matrix must have same shape to add")
        return Matrix([[self._matrix[i][j] + m.matrix[i][j] for j in range(len(self._matrix[0]))] for i in range(len(self._matrix))])
    
    def mul(self, m: "Matrix"):
        """
        return Matrix object of the mulitplication of the current matrix with another matrix

        reminder: in matrix multiplication, order is capital:

        [[1,2;3]        [[10,11]              [[(1*10+2*12+3*14), (1*11+2*13+3*15)]
         [4,5,6]]   *    [12,13]      =        [(4*10+5*12+6*14), (4*11+5*13+6*15)]]
                         [14,15]]

        [[10,11]        [[1,2,3]              [[(10*1+11*4), (10*2+11*5), (10*3+11*6)]
         [12,13]    *    [4,5,6]]     =        [(12*1+13*4), (12*2+13*5), (12*3+13*6)]
         [14,15]]                              [(14*1+15*4), (14*2+15*5), (14*3+15*6)]]
        """
        shape1 = self.shape
        shape2 = m.shape
        if shape1[1] != shape2[0]:
            raise AssertionError(f"matrix of shape {self.shape} cannot multiply with matrix of shape {shape2}")

        rowsA = len(self._matrix)
        colsB = len(m.matrix[0])
        result = [[0] * colsB for i in range(rowsA)]

        for i in range(rowsA):
            # iterating by column by B
            for j in range(colsB):
                # iterating by rows of B
                for k in range(len(m.matrix)):
                    result[i][j] += self._matrix[i][k] * m.matrix[k][j]

        return Matrix(result)

