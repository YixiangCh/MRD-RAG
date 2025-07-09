import os
import re
import json
import time

from utils import *
from tqdm import tqdm
from langchain_community.graphs import Neo4jGraph
from concurrent.futures import ProcessPoolExecutor, as_completed


def divide_into_batches(arr, batch_size):
    batches = []
    num_batches = len(arr) // batch_size
    for i in range(num_batches):
        batch = arr[i*batch_size:(i+1)*batch_size]
        batches.append(batch)
    if len(arr) % batch_size != 0:
        last_batch = arr[num_batches*batch_size:]
        batches.append(last_batch)
    return batches

def process_items_with_pool(trees,batch_size):
    batches = divide_into_batches(trees,batch_size=batch_size)

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(store2neo4j, batch) for batch in batches]
        for future in tqdm(as_completed(futures), total=len(batches), desc="Processing items"):
            future.result()

def store2neo4j(batch):
    NEO4J_URI = "bolt://localhost:7687"
    NEO4J_USERNAME = "neo4j"
    NEO4J_PASSWORD = "12345678"
    NEO4J_DATABASE = "modern-medicine-disease"
    # NEO4J_DATABASE = "tcm-syndrome"
    source =  "yixue" # The data source (e.g. yixue)
    main_node_label = "Disease"
    
    kg = Neo4jGraph(url=NEO4J_URI, username=NEO4J_USERNAME, password=NEO4J_PASSWORD, database=NEO4J_DATABASE)

    for tree_dict in batch:
        tree = TreeNode.from_dict(tree_dict)
        name = tree.value # disease name

        main_node_params = {"name":name}
        main_node_uuid = create_node(kg=kg,node_label=main_node_label,params=main_node_params)
        
        info_tree_root_params = {"source":source}
        info_tree_root_uuid = create_node(kg=kg,node_label="Info Tree Root",params=info_tree_root_params)

        create_relationship(kg=kg,node_id_1=main_node_uuid,node_id_2=info_tree_root_uuid,relationship_label="main_node-tree_root")

        embedding_root_node_uuid = create_node(kg=kg,node_label="Embedding Root")
        create_relationship(kg=kg,node_id_1=main_node_uuid,node_id_2=embedding_root_node_uuid,relationship_label="main_node-embedding_root")

        info_tree_node_uuid = store_info_tree(kg=kg,root=tree)
        create_relationship(kg=kg,node_id_1=info_tree_root_uuid,node_id_2=info_tree_node_uuid,relationship_label="tree_root-tree_node")

if __name__=="__main__":
    script_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_path)

    # neo4j param should be set in `store2neo4j` function again.
    NEO4J_URI = "bolt://localhost:7687"
    NEO4J_USERNAME = "neo4j"
    NEO4J_PASSWORD = "12345678"
    NEO4J_DATABASE = "modern-medicine-disease"
    # NEO4J_DATABASE = "tcm-syndrome"
    source =  "yixue" # The data source (e.g. yixue)
    main_node_label = "Disease"
    
    kg = Neo4jGraph(url=NEO4J_URI, username=NEO4J_USERNAME, password=NEO4J_PASSWORD, database=NEO4J_DATABASE)

    # Clear the original data
    delete_cypher = """MATCH (n)
    OPTIONAL MATCH (n)-[r]-()
    DELETE n,r"""
    kg.query(delete_cypher)
    

    tree_file_name = "diseases.json"
    tree_file_path = os.path.join(script_dir,tree_file_name)
    with open(tree_file_path,mode="r",encoding="utf-8") as file:
        trees = json.load(file)

    # Multi-process mode
    batch_size = 100
    process_items_with_pool(trees,batch_size)
    
# You can use cypher "MATCH path=(n:`Disease` {name: '肺癌'})-[*]-(m) RETURN path" to check if the disease has been stored in to the database.