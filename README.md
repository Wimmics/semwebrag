# semwebrag
* This is our own approach of a GraphRAG pipeline, based on knowledge graphs. *

---


## Launching the projet with graphs already built

You only need to use this two commands : 

```bash
python install.py
python serverLauncher.py
```


You should see in http://localhost:8000/interface.html an interface in wich you can select your domain and query the GraphRAG ! 

## Build your own graph : 

You will need to update at least these lines in the file config.ini  : 

- domain : folder in wich you want to store the indexes and the graph.
- text_file_path : path to your textual datafile.
- ontology_path : path to your ontology (.tll)

You can then use the *build_knowledge_graph_aligned_with_ontology* function in mainPipeline.py (there is a commented exemple at the end of the file)

Then the function *process_query* will generate a prompt to give to an LLM in a file named query_enrichie.txt