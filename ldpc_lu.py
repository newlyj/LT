import numpy as np
import scipy

def build_H(n, d_v, d_c, seed=None):
    """
    Builds a regular Parity-Check Matrix H (n, d_v, d_c) following Callager's algorithm.

    Parameters:

     n: Number of columns (Same as number of coding bits)
     d_v: number of ones per column (number of parity-check equations including a certain variable)
     d_c: number of ones per row (number of variables participating in a certain parity-check equation);
    ---------------------------------------------------------------------------------------

     Returns: Check Matrix H 

    """
    rnd = np.random.RandomState(seed)
    if n % d_c:
        raise ValueError("""d_c must divide n.""")

    if d_c <= d_v:
        raise ValueError("""d_c must be greater than d_v.""")
    
    # Compute the number of columns and m is equal to the number of check bit
    m = (n * d_v) // d_c

    # Compute the basic matrix H0
    Set = np.zeros((m//d_v, n), dtype=int)
    a = m // d_v

    # Filling the first set with consecutive ones in each row of the set
    for i in range(a):
        for j in range(i * d_c, (i+1)* d_c):
            Set[i, j] = 1
    # Create list of Sets and append the first reference set
    Sets = []
    Sets.append(Set.tolist())

    # reate remaining sets by permutations of the first set's columns:
    for i in range(1, d_v):
        newSet = rnd.permutation(np.transpose(Set)).T.tolist()
        Sets.append(newSet)

    # Returns concatenated list of sest:
    H = np.concatenate(Sets)
    return H

def build_G(H):
    """
    Builds a regular Parity-Check Matrix H (n, d_v, d_c) based on the Gaussian elimination method .

    Parameters:

     H: a regular Parity-Check Matrix H (n, d_v, d_c)
    ---------------------------------------------------------------------------------------

     Returns: Check Matrix H transformed by a determinant and Generation Matrix G

    """
    m, n = H.shape
    for i in range(m):
        if H[i,i]==0:
            if np.where(H[i,:]==1)[0].size != 0:
                index = np.where(H[i,:]==1)[0][0]
                H[:,[i,index]] = H[:,[index,i]]
        for k in range(i+1,m):
            if H[k,i] == 1:
                H[k,:] = H[i,:] + H[k,:]
                H[k,:] = np.mod(H[k,:], 2)
    #print(H)
    for i in range(m-1,0,-1):
        for k in range(i-1,-1,-1):
            if H[k,i]==1:
                H[k,:] = H[i,:] + H[k,:]
                H[k,:] = np.mod(H[k,:], 2)
    #print(H)

    PP = H[:,m:n]
    #print(PP)
    a = np.transpose(PP)
    b = np.diag([1] * (n-m))
    G = np.hstack((a,b))

    return H, G

def binaryproduct(X, Y):
    """
    Binary Matrix Product

    """
    A = X.dot(Y)
    return A % 2

def encoder(G, x):
    """
    Encoder

    Parameters:

     G: Generation matrix
     x: transmit symbols before coding
    ---------------------------------------------------------------------------------------

     Returns: The coding symbols by LDPC codec

    """
    y = binaryproduct(x, G)
    return y

def BPSK(y, snr, seed=None):
    """
    BPSK modulation

    Parameters:

     snr: Signal-to-noise ratio 
     y: transmit symbols 
    ---------------------------------------------------------------------------------------

     Returns: The symbols at receiver side

    """
    y = (-1) ** y

    sigma = 10 ** (- snr / 20)
    n = y.shape[0]
    rnd = np.random.RandomState(seed)
    e = rnd.randn(n) * sigma

    z = y + e
    return z


def Bits2i(H, i):
    """
    Computes list of elements of N(i)-j:
    List of variables (bits) connected to Parity node i.

    """
    m, n = H.shape
    return ([a for a in range(n) if H[i, a]])


def Nodes2j(tH, j):
    """
    Computes list of elements of M(j):
    List of nodes (PC equations) connecting variable j.

    """

    return Bits2i(tH, j)

def BitsAndNodes(H):
    m, n = H.shape
    tH = np.transpose(H)

    Bits = [Bits2i(H, i) for i in range(m)]
    Nodes = [Nodes2j(tH, j) for j in range(n)]

    return Bits, Nodes


def belief_pro_LDPC(symbol, H, max_iter=1):
    """
    A LDPC decoder based on the belief propagation method .

    Parameters:

    symbol: received symbols
    H : check matrix 
    ---------------------------------------------------------------------------------------

     Returns: decoded message bit

    """
    # row : check bit length m
    # col : coding bit length n
    row, col = np.shape(H)
    # Compute the message bit length k
    k = col - row

    # Initial
    beta = np.zeros([row, col], dtype=float)
    alpha = np.zeros([row, col])
    decide = np.zeros(col)
    m = np.zeros(len(symbol))
    prod = np.prod
    tanh = np.tanh
    atanh = np.arctanh
    count = 0

    # find the nonzero element
    BitsNodesTuple = BitsAndNodes(H)
    Bits = BitsNodesTuple[0]  # Nm
    Nodes = BitsNodesTuple[1]  # Mn

    for check_id in range(row):
        Ni = Bits[check_id]
        for bit_id in Ni:
            beta[check_id][bit_id] = symbol[bit_id]  # eq.(4) v_bit -> check
    # message updata
    while (True):
        count += 1
        # Step 2 Horizontale
        for bit_id in range(col):  # for each bits node
            Mi = Nodes[bit_id]  # lists check nodes of bit node i
            for check_id in Mi:
                Ni = Bits[check_id]  # lists bit nodes of check node j
                Nij = Ni[:]
                if bit_id in Ni:
                    Nij.remove(bit_id)
                X = prod(tanh(0.5 * beta[check_id, Nij]))
                alpha[check_id][bit_id] = 2 * atanh(X)  # w_check -> bit

            # Step 2 Verticale
            Mij = Mi[:]
            for check_id in Mi:
                Mij.remove(check_id)
                beta[check_id][bit_id] = symbol[bit_id] + sum(alpha[Mij, bit_id])
        # Step 3
        for bit_id in range(col):  # for check node
            Ni = Nodes[bit_id]
            m[bit_id] = sum(alpha[Ni, bit_id]) + sum(beta[Ni, bit_id])  # eq.(4) v_bit -> check

        for i in range(col):  # for early bit node
            Mi = Nodes[i]  # lists check nodes of bit node i
            decide[i] = symbol[i] + sum(alpha[Mi, i])

        # End condition
        if count >= max_iter:
            break

    # Soft decision
    decode_LDPC = np.zeros(k)
    for i in range(k):
       if decide[i+row] < 0:
           decode_LDPC[i] = 1
    return decode_LDPC