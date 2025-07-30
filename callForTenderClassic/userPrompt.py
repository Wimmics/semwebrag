import callForTenderClassic.pipelineLinkerMC
import os
from sys import argv

query = argv[1]
nChunks = int(argv[2])

print ("nchunks = ", nChunks)


callForTenderClassic.pipelineLinkerMC.process_query(query,"callForTenderClassic/outputLinkerLinked.ttl" , neighborChunks=nChunks )#, embeddings) 