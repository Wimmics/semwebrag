import financeClassic.pipelineLinkerFC
import os
from sys import argv

query = argv[1]
nChunks = int(argv[2])

print ("nchunks = ", nChunks)


financeClassic.pipelineLinkerFC.process_query(query,"financeClassic/outputLinkerLinked.ttl" , neighborChunks=nChunks )#, embeddings) 