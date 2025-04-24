import finance.pipelineLinkerF
import os
from sys import argv

query = argv[1]
nChunks = int(argv[2])

print ("nchunks = ", nChunks)


finance.pipelineLinkerF.process_query(query,"finance/outputLinkerLinked.ttl" , neighborChunks=nChunks )#, embeddings) 