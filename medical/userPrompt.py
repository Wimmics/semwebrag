import medical.pipelineLinkerM
import os
from sys import argv
# from langchain.embeddings import HuggingFaceEmbeddings


# embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

query = argv[1]

nChunks = int(argv[2])


medical.pipelineLinkerM.process_query(query,"medical/outputLinkerLinkedM.ttl", neighborChunks=nChunks)#, embeddings) 