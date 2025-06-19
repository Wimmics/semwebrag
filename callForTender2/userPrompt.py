import callForTender2.pipelineLinkerSOLIHA
import os
from sys import argv

query = argv[1]
nChunks = int(argv[2])

print ("nchunks = ", nChunks)


callForTender2.pipelineLinkerSOLIHA.process_query(query,"callForTender2/outputLinkerLinked.ttl" , neighborChunks=nChunks )#, embeddings) 