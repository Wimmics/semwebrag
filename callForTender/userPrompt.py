import callForTender.pipelineLinkerSOLIHA
import os
from sys import argv

query = argv[1]
nChunks = int(argv[2])

print ("nchunks = ", nChunks)


callForTender.pipelineLinkerSOLIHA.process_query(query,"callForTender/outputLinkerLinked.ttl" , neighborChunks=nChunks )#, embeddings) 