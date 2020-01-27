from encoder import encode
from decoder import decode
from lt_bpdecoder import lt_bpdecoder
import ldpc_lu as l
import numpy as np

# Initial
n = 500
redundancy = 2
# Generate test bit
file_blocks = np.random.randint(2, size=n)
#print("The message bit: \n", file_blocks)
file_blocks_n = len(file_blocks)

# The redundancy setting
drops_quantity = int(file_blocks_n * redundancy)
for snr in range(5,10):

    file_symbols = []
    data = np.zeros([drops_quantity])
    i=0
    for curr_symbol in encode(file_blocks, drops_quantity=drops_quantity):
        file_symbols.append(curr_symbol)
        data[i] = curr_symbol.data
        i = i + 1
    #print("The code bit: \n", data)

    # BPSK
    z = l.BPSK(data,snr)
    #print(z)

    receiver = np.array(z <= 0).astype(int)
    #print(receive)

    for i in range(len(file_symbols)):
        file_symbols[i].data = receiver[i]

    # Recovering the blocks from symbols
    recovered_blocks, recovered_n = decode(file_symbols, blocks_quantity=file_blocks_n)
    #print("The decoded bit: \n", recovered_blocks)
    print("When SNR is: ",snr ,"The MSE is : \n", sum(abs(recovered_blocks-file_blocks)))
