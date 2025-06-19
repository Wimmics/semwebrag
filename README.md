# semwebrag



## Domains 

- **finance** 
Graph built from a part of financialQA. 
Ontology used : FIBO (dev.fibo-quickstart.ttl)
- **medical** 
Graph built from a part of covidQA.
Ontology used : ATC (ATC.ttl)
- **callForTender & callForTender2** 
Graph built from private call for tender datas.
Ontology used : Public Contracts Ontology (public-contracts.ttl). 
There is a difference of chunking methods used between the two domains
- **financeClassic** 
No graph built, we just do a simple RAG with the same datas than fiance domain in order to compare it with the GraphRAG method.

## To build the graph and the embeddings database :

- Go into the folder of the domain of your choice and open the file named pipelineLinker*
- At the end of it decomment the line calling the build_knowledge_graph_aligned_with_ontology fonction.
- You can now build the graph from the root of the project by using the commande python -m {domain}.{pipelineLinkerfile}

examples : 

```bash
python -m finance.pipelineLinkerF
python -m callforTender.pipelineLinkerSOLIHA
```

## To see the GraphRAG in action 

- Be sure to **re-comment** the pipelineLinker files of the domains in wich you build the graph

- Lauch these commands : 

```bash
python -m http.server

python appflask.py
```


You should see in http://localhost:8000/interface.html an interface in wich you can select your domain and query the graphRAG ! 

If you want to use the classic RAG, you can find it in http://localhost:8000/interfaceClassic.html