import os
import re
import json
import time

from utils import *
from tqdm import tqdm
from langchain_community.graphs import Neo4jGraph
from langchain_text_splitters import RecursiveCharacterTextSplitter

def extract_list(llm_result):
    pattern = r'\[.*?\]'
    matches = re.findall(pattern, llm_result, re.DOTALL)
    result = []
    try:
        for match in matches:
            match = json.loads(match)
            for item in match:
                if isinstance(item,str):
                    result.append(item)
    except Exception as e:
        print(e)
    return result

def get_prompt_pseudo_medical_history(name,disease_info):
    sys_content = f"""我将给你一些关于“{name}”的基本信息，比如其临床表现、症状等。请你根据此信息，生成关于该病的多个病历，主要包括患者主诉，现病史，既往史，所做检查等信息。生成的病历要求如下：
    1. 生成2至5个病历。所给的疾病基本信息越多，生成的病历个数越多。
    2. 单个病历应当内容完整，包括患者主诉，现病史，既往史等信息；还应该逻辑清晰；此外最好是能够描述出全部的症状。
    3. 不同的病历之间应当具有多样性，但需要用到所提供的关于该疾病的基本信息，不能编造症状。
    4. 返回的格式要求：用单个字符串来表示单个病历，所有的病历描述保存在一个数组中，例如：["患者1：主诉：...。现病史：...。既往史：...。所做检查：...。诊断：...。", "患者2：主诉：...。现病史：...。所做检查：...。诊断：...。"]。
"""

    input_ = f"""以下为关于“{name}”的基本信息：
    {disease_info}\n\n生成的病历：\n"""
        
    prompt = [
        {"role": "system", "content": sys_content},
        {"role": "user", "content": input_}
    ]
    return prompt

def get_pseudo_medical_history(name,info,chat_args):
    prompt = get_prompt_pseudo_medical_history(name=name,disease_info=info)
    llm_result = run_llm_retry(prompt=prompt,llm_args=chat_args)
    if not llm_result:
        print(f"The request about {name} failed too many times!!!")
        exit(0)
    pseudo_medical_histories = extract_list(llm_result) # list
    if not pseudo_medical_histories:
        print(f"The request about {name} doesn't follow the format requirement!!!")
        return [info]
    return pseudo_medical_histories

def store_embedding(kg,node_uuid,property_name,embedding):
    set_embedding_cypher = """MATCH (n)\nWHERE elementid(n)=\"%s\"\nCALL db.create.setNodeVectorProperty(n,\"%s\",$embedding)"""%(node_uuid,property_name)
    kg.query(set_embedding_cypher,params={"embedding":embedding})

def get_children_id(kg,parent_uuid,relationship_label=None,children_label=None):
    relationship_label_str = ":`%s`"%relationship_label if relationship_label else ""
    children_label_str = ":`%s`"%children_label if children_label else ""
    cypher = """MATCH (parent)-[r%s]->(children%s)\nWHERE elementid(parent)=\"%s\"\nRETURN elementid(children) as children_id"""%(relationship_label_str,children_label_str,parent_uuid)
    query_result = kg.query(cypher)
    result = [children_id["children_id"] for children_id in query_result]
    return result

def check_embedding_node(kg,embedding_root_uuid,embedding_node_label,indexing_type):
    check_cypher = """MATCH (embedding_root:`Embedding Root`)-[r]->(embedding_node:`%s`)\nWHERE elementid(embedding_root)=\"%s\" AND embedding_node.embedding IS NOT NULL AND embedding_node.embedding_type=\"%s\"\nRETURN count(embedding_node) AS count_node"""%(embedding_node_label,embedding_root_uuid,indexing_type)
    query_result = kg.query(check_cypher)
    result = query_result[0]["count_node"]
    return result

if __name__=="__main__":
    script_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_path)

    NEO4J_URI = "bolt://localhost:7687"
    NEO4J_USERNAME = "neo4j"
    NEO4J_PASSWORD = "12345678"
    NEO4J_DATABASE = "modern-medicine-disease"
    # NEO4J_DATABASE = "tcm-syndrome"

    source =  "yixue"
    main_node_label = "Disease"
    indexing_type = "pseudo medical history"
    # indexing_type = "diagnosis info"

    if source=="dayi":
        search_list = ["症状","病因","表现"]
    elif source=="yixue":
        search_list = ["表现","症状","诊断","检查"]
    
    need_desc = True

    embedding_model_name = "text-embedding-3-small"  
    chat_model_name = "gpt-4o-mini"  

    embedding_args = {
    "openai_api_base": "openai_api_base",
    "openai_api_key": "openai_api_key",
    "model": embedding_model_name,
    }

    chat_args = {
    "openai_api_base": "openai_api_base",
    "openai_api_key": "openai_api_key",
    "model": chat_model_name,
    "temperature": 0,
    "max_tokens": 2048,
    "presence_penalty": 0
    }

    separators = ["\n",".","．","。","?","？","!","！",";","；",":","：",",","，","、"," ","\u200B",""]
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        model_name=chat_model_name,
        separators=separators,
        chunk_size=8192,
        chunk_overlap=0
        )

    kg = Neo4jGraph(url=NEO4J_URI, username=NEO4J_USERNAME, password=NEO4J_PASSWORD, database=NEO4J_DATABASE)
    embedding_node_label = indexing_type + " " + "Embedding Node"
    create_index_cypher = """CREATE VECTOR INDEX `%s` IF NOT EXISTS\nFOR (n:`%s`) ON (n.%s)\nOPTIONS { indexConfig: {\n`vector.dimensions`: %d,\n`vector.similarity_function`: 'cosine'\n}}"""%(indexing_type,embedding_node_label,"embedding",1536)
    kg.query(create_index_cypher)

    find_all_cypher = """MATCH (n:`%s`)\nRETURN elementid(n) AS node_id"""%main_node_label
    main_node_uuids = kg.query(find_all_cypher)


    for idx,item in tqdm(enumerate(main_node_uuids)):
        main_node_id = item["node_id"]
        embedding_root_uuid = get_children_id(kg=kg,parent_uuid=main_node_id,children_label="Embedding Root")[0]
        if check_embedding_node(kg=kg,embedding_root_uuid=embedding_root_uuid,embedding_node_label=embedding_node_label,indexing_type=indexing_type):
            continue
        info_tree_root_node_uuid = get_children_id(kg=kg,parent_uuid=main_node_id,children_label="Info Tree Root")[0]
        info_tree_first_node_uuid = get_children_id(kg=kg,parent_uuid=info_tree_root_node_uuid,children_label="Info Tree Node")[0]
        info_tree = restore_info_tree(kg=kg,root_node_uuid=info_tree_first_node_uuid)
        name = info_tree.value

        info = get_info(info_tree,splitter,search_list=search_list,need_desc=need_desc)
        if indexing_type == "pseudo medical history":
            embedding_texts = get_pseudo_medical_history(name,info,chat_args)
        else:
            embedding_texts = [info]
        embeddings = get_embeddings(inputs=embedding_texts,args=embedding_args)
        if embeddings:
            for embedding_text,embedding in zip(embedding_texts,embeddings):
                try:
                    # The following three steps should be an atomic operation.
                    embedding_node_uuid = create_node(kg,node_label=[embedding_node_label,"Embedding Node"],params={"embedding_type":indexing_type,"source":source,"text":embedding_text,"name":name})
                    store_embedding(kg=kg,node_uuid=embedding_node_uuid,property_name="embedding",embedding=embedding)
                    create_relationship(kg=kg,node_id_1=embedding_root_uuid,node_id_2=embedding_node_uuid,relationship_label="embedding_root-embedding_node")
                except Exception as e:
                    print(e)

