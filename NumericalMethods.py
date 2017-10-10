import numpy as np;

# A well conditioned nxn matrix is a representation of an nxn space so 
# the n vectors must not to be paralell between them in any combination.
# This algorithm just measures the degree of paralellism between the set
# of vectors in a range [0, 1]. If all the vector are orthogonal then 
# the result is 0 if there is a paralellism the result is 1.
# Values near or equal to 0 tells us the matrix is well conditioned but 
# values near or equal to 1 tells us about an ill conditioned matrix
def MatrixCondition(A):

    Sum = np.zeros(A.shape[0] - 1)
        
    for i in xrange(A.shape[0] - 1):

        l1 = np.sqrt(A[i].dot(A[i]))

        if l1 == 0.0:
            return 1

        PartialSum = 0
        
        for j in xrange(i+1, A.shape[0]):

            l2 = np.sqrt(A[j].dot(A[j]))
            
            if l2 == 0.0:
                return 1

            Cos =  np.abs( (A[i].dot(A[j]))/(l1*l2) )

            PartialSum += Cos

        Sum[i]  = PartialSum / j

    return np.sum(Sum) / Sum.shape

# Gaussian elimination with partial pivoting i.e search the row with the 
# max value in its first column and swap this row with the first one
def GaussElimination(A, b):
    
    if A.shape[0] != A.shape[1] or A.shape[0] != b.shape[0]:
        return A

    A = np.concatenate((A, np.array([b]).T), axis=1);

    MaxValue = np.abs(A[0][0]);
    MaxRowIdx = 0;
    
    for i in xrange(1, A.shape[0]):
        if np.abs(A[i][0]) > MaxValue:
            MaxValue = np.abs(A[i][0]);
            MaxRowIdx = i;

    # Do the partial pivoting
    if MaxRowIdx != 0:
        HelperRow = A[0].copy();
        A[0] = A[MaxRowIdx];
        A[MaxRowIdx] = HelperRow;

    # Let i=1,...,N
    # Let j=1,...,N
    # Let Eq_i be the ith equation
    # Let c_ij be the ij coeficient in Eq_i
    # elimination phase: Eq_i = Eq_i / c_i. Then Eq_(i+1) = Eq_(i+1) - c_ik * Eq_i, with k=i,...,N
    for i in xrange(A.shape[0]):
        if A[i][i] == 0:
            raise Exception("Singular Matrix");

        A[i] = A[i] / float(A[i][i]); 

        for j in xrange(i+1, A.shape[0]):
            A[j] -= (A[j][i]  * A[i]);

    # At this point the matrix is triangular 
    # so now we can perform back substitution
    for i in xrange(A.shape[0]-1, -1, -1):
    
        for j in xrange(i):
            A[j] -= (A[j][i] * A[i]);

    # return the last colum transposed i.e x1, x2, x3
    return A[:,[A.shape[0]]].T;


# A = np.array([
#     [ 4.0, -2.0,  1.0],
#     [-2.0,  4.0, -2.0],
#     [ 1.0, -2.0,  4.0]])

# A = np.array([
#     [ 0.143,  0.357,  2.010],
#     [-1.310,  0.911,  1.990],
#     [ 0, -4.300, 0.605]])

# A = np.array([
#     [ 0.143,  0.357,  2.010],
#     [-1.310,  0.911,  1.990],
#     [ 11.20, -4.300, -0.605]])

# A = np.array([
#     [2,3,-7],
#     [5,4,-2],
#     [7,-3,6]
# ])

# A = np.array([
#     [400, -201],
#     [-800, 401]
# ])

A = np.array([
    [1,0,0],
    [0,1,0],
    [0,1,1]
])

print MatrixCondition(A)
print np.linalg.cond(A)
#b = np.array([-5.173, -5.458, 4.415]);
#A = GaussElimination(A, b);

