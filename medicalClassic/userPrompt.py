import medicalClassic.pipelineLinkerMC
import os
from sys import argv

query = argv[1]
nChunks = int(argv[2])

print ("nchunks = ", nChunks)


medicalClassic.pipelineLinkerMC.process_query(query,"medicalClassic/outputLinkerLinked.ttl" , neighborChunks=nChunks )#, embeddings) 