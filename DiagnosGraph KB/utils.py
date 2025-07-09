#coding=utf-8

import os
import json
import requests
import tiktoken
import argparse

from openai import OpenAI

class TreeNode:
    def __init__(self, value=None, node_type=None):
        self.value = value
        self.node_type = node_type
        self.children = {}

    def add_child(self, child):
        if child.value not in self.children:
            self.children[child.value] = child
        return self.children[child.value]

    def to_dict(self):
        if self.value is None:  # Special handling of root nodes
            return [child.to_dict() for child in self.children.values()]
        if not self.children:  # Leaf node
            return {"text": self.value, "node_type": self.node_type}
        # Intermediate node
        return {
            self.value: [child.to_dict() for child in self.children.values()],
            "node_type": self.node_type
        }

    @staticmethod
    def from_dict(data, parent=None):
        if isinstance(data, dict):
            if 'text' in data:
                return TreeNode(data['text'],data['node_type'])
            for key, value in data.items():
                node = TreeNode(key,data['node_type'])
                for child in value:
                    node.add_child(TreeNode.from_dict(child, node))
                return node
        elif isinstance(data, list):
            root = TreeNode() if parent is None else parent
            for item in data:
                root.add_child(TreeNode.from_dict(item, root))
            return root

    def get_subtree(self, value):
        if self.value == value and self.node_type in ["h2","h3","h4"]:
            return self
        for child in self.children.values():
            result = child.get_subtree(value)
            if result:
                return result
        return None
    
    def get_subtree_fuzzy(self, value): # Returns the title subtree when the given value is included in the title
        if value in self.value and self.node_type in ["h2","h3","h4"]:
            return self
        for child in self.children.values():
            result = child.get_subtree_fuzzy(value)
            if result:
                return result
        return None

    def get_subtrees_fuzzy_by_values(self, values):  # Finds the largest list of multiple subtrees based on the values provided
        
        subtrees = [self.get_subtree_fuzzy(value) for value in values]
        
        def is_subtree(parent, child):
            # Check whether a subtree is a subtree of another
            if not parent or not child:
                return False
            if parent == child:
                return True
            for c in parent.children.values():
                if is_subtree(c, child):
                    return True
            return False
        
        # Remove a subtree that is a subtree of another subtree
        final_subtrees = []
        for i,subtree in enumerate(subtrees):
            flag = True
            for other in subtrees: # If there is other (any subtree in the subtrees list other than subtree) such that the subtree is a subtree of Other, the subtree is not added to the final subtree list
                if other != subtree and other:
                    if is_subtree(other,subtree):  
                        flag = False
                        break
            if flag:
                final_subtrees.append((values[i],subtree))
            else:
                final_subtrees.append((values[i],None))
        
        return final_subtrees

    def get_all_intermediate_nodes(self):
        nodes = []
        if self.children:  # Non-leaf node
            nodes.append((self.value,self.node_type))
        for child in self.children.values():
            nodes.extend(child.get_all_intermediate_nodes())
        return nodes
    
    def tree2str_list(self):
        str_list = []

        repeat_num = int(self.node_type[-1]) if self.node_type[-1].isdigit() else 0 # h1 has one '#' prefix,h2 has two '#' prefixes, and so on
        prefix = "#"*repeat_num
        str_list.append(f"{prefix} "+self.value)
        for child in self.children.values():
            str_list.extend(child.tree2str_list())
        return str_list
    
    def tree2str(self):
        return "\n".join(self.tree2str_list())

def store_info_node(kg, cur_node, parent_uuid=None,node_order=0):
    params={"value":cur_node.value,"node_type":cur_node.node_type}
    cur_node_uuid = create_node(kg=kg,node_label="Info Tree Node",params=params)

    # If a parent node exists, create a parent-child relationship and add an order attribute
    if parent_uuid is not None:
        create_relationship(kg=kg,node_id_1=parent_uuid,node_id_2=cur_node_uuid,relationship_label="tree_node-tree_node",params={"node_order":node_order})

    # Recursively do the same for each child of the current node
    for child_node_order,child in enumerate(cur_node.children.values()):
        store_info_node(kg, child, cur_node_uuid, child_node_order)
    return cur_node_uuid

def store_info_tree(kg,root):
    root_node_uuid = store_info_node(kg, root)
    return root_node_uuid

def restore_info_tree_node(kg,cur_node_uuid,cur_node):
    cypher = """MATCH (parent)-[r]->(child)\nWHERE elementid(parent)=\"%s\"\nRETURN child,elementid(child) AS child_id\nORDER BY toInteger(r.node_order)"""%cur_node_uuid
    query_result = kg.query(cypher)
    for child in query_result:
        child_value = child["child"]["value"]
        child_node_type = child["child"]["node_type"]
        child_id = child["child_id"]
        child_node = TreeNode(value=child_value,node_type=child_node_type)
        cur_node.add_child(child=child_node)
        restore_info_tree_node(kg,child_id,child_node)

def restore_info_tree(kg,root_node_uuid): # Gets the child tree based on the root node uuid
    cypher = """MATCH (n)\nWHERE elementid(n) = \"%s\"\nRETURN n"""%root_node_uuid
    query_result = kg.query(cypher)
    root_value = query_result[0]["n"]["value"]
    root_node_type = query_result[0]["n"]["node_type"]
    root = TreeNode(value=root_value,node_type=root_node_type)
    restore_info_tree_node(kg,root_node_uuid,root)
    return root

def dict_to_custom_string(d):
    custom_str = "{"
    for key, value in d.items():
        if isinstance(value,str):
            value = value.replace("\\","\\\\").replace('"','\\"')
        custom_str += f"{key}: \"{value}\", "
    custom_str = custom_str.rstrip(", ") + "}"
    return custom_str

def create_node(kg,node_label,params=None):
    params_str = dict_to_custom_string(params) if params else ""
    if isinstance(node_label,list):
        node_label_str = "".join([f":`{label}`" for label in node_label])
    elif isinstance(node_label,str):
        node_label_str = f":`{node_label}`"
    cypher = """CREATE (n%s%s)\nRETURN elementid(n) as node_id"""%(node_label_str,params_str)
    query_result = kg.query(cypher)
    node_uuid = query_result[0]["node_id"]
    return node_uuid

def delete_node(kg,node_id):
    cypher = """MATCH (n)\nWHERE elementid(%s)\nDETACH DELETE n"""%(node_id)
    query_result = kg.query(cypher)
    
    return

def create_relationship(kg,node_id_1,node_id_2,relationship_label,params=None):
    params_str = dict_to_custom_string(params) if params else ""
    cypher = """MATCH (n_1),(n_2)\nWHERE elementid(n_1)=\"%s\" AND elementid(n_2)=\"%s\"\nMERGE (n_1)-[r:`%s`%s]->(n_2)"""%(node_id_1,node_id_2,relationship_label,params_str)
    kg.query(cypher)

def get_info(info_tree,splitter,search_list=[],need_desc=False):
    name = info_tree.value

    info = ""
    if search_list:
        search_results = info_tree.get_subtrees_fuzzy_by_values(search_list)
        for search_result in search_results:
            search_value, search_tree = search_result
            search_result_str = search_tree.tree2str() if search_tree else "" 
            if search_result_str:
                info += search_result_str + "\n\n"
        if need_desc:
            desc = ""
            for child in info_tree.children.values():
                if child.node_type == "leaf":
                    desc = desc + child.value + "\n\n"
            info =  desc + info

    if not info:
        for child in info_tree.children.values():
            if child.node_type == "leaf":
                info = child.value + "\n\n" + info
        if not info:
            info = info_tree.tree2str()
    
    info = splitter.split_text(info)[0]

    if not info:
        print(f"{name}没有信息！")

    return info


def get_patient_infos(test_data_file_path):
    with open(test_data_file_path,mode="r",encoding="utf-8") as file:
        data_list = json.load(file)
    return data_list

def get_patient_prompt(history, patient_info):
    disease_name = patient_info["disease name"]
    patient_info_str = patient_info["patient info"]
    sys_content = """请你扮演一位“%s”患者，根据以往的对话历史，对医生进行回复，要求如下：\n1. 你的语气应当平和、友好，符合患者的说话风格。\n2. 本次说话你可以透露一部分新的医疗信息给医生，例如自身症状、医学检查结果（不含诊断结果）、医学指标等。若医生询问你自身的医疗信息，你应该如实提供。\n3. 特别需要注意的是：不能将你的医疗信息一次性完全透露给医生，除非医生已经指出你所患有的疾病，否则不能主动说出自己所患有的疾病名称（即“%s”）！\n4. 本次说话时不可以超过30个字，不要说与问诊无关的话。\n以下是你的信息：\n%s\n"""%(disease_name, disease_name, patient_info_str)
    history_str = ""
    for i,utterance in enumerate(history):
        role = f"医生" if utterance["role"]=="doctor" else "患者"
        history_str += role + "：" + utterance["content"] + "\n"
    history_str = history_str + f"患者："
    prompt = [{"role": "system", "content": sys_content},{"role": "user", "content": history_str}]
    return prompt

def Patient(round,patient_info,history,llm_args):
    first_question = patient_info["first_question"]
    if round==0 and first_question:
        patient_llm_response = first_question
    else:
        patient_prompt = get_patient_prompt(history=history,patient_info=patient_info)
        patient_llm_response = run_llm_retry(prompt=patient_prompt,llm_args=llm_args)
        if not patient_llm_response:
            return None
    return patient_llm_response

def get_num_tokens(string, model_name="gpt-3.5-turbo"):
    encoding = tiktoken.encoding_for_model(model_name=model_name)
    encode = encoding.encode(string)
    return len(encode)

def run_llm(prompt, args):
    client = OpenAI(
        api_key=args["openai_api_key"],
        base_url=args["openai_api_base"],
    )
    try:
        chat_response = client.chat.completions.create(
            model=args["model"],
            messages=prompt,
            temperature=args["temperature"],
            presence_penalty=args["presence_penalty"],
            max_tokens=args["max_tokens"],
            timeout=500,
        )
        llm_result = chat_response.choices[0].message.content
        return llm_result
    except Exception as e:
        print(e)
        return None
    
def run_llm_retry(prompt,llm_args,request_time_patience=3):
    request_time = 0
    while True:
        llm_result = run_llm(prompt=prompt,args=llm_args)
        request_time += 1
        if llm_result:
            return llm_result
        if request_time>=request_time_patience:
            model = llm_args["model"]
            print(f"{model} request failed!")
            return None

def get_embeddings(inputs,args):
    headers = { 
    "Content-Type": "application/json", 
    "Authorization": "Bearer "+args["openai_api_key"]
    } 
    data = { 
    "input": inputs,
    "encoding_format": "float",
    "model": args["model"]
    } 
    try:
        response = requests.post(args["openai_api_base"], headers=headers, data=json.dumps(data)) 
        response_json = response.json()
        embeddings = []
        for data in response_json["data"]:
            embeddings.append(data["embedding"])
        if len(inputs)==len(embeddings):
            return embeddings
        else:
            return None
    except Exception as e:
        print(e)
        return None

def get_embeddings_retry(inputs,args,request_time_patience=5):
    request_time = 0
    while True:
        embeddings = get_embeddings(inputs,args)
        request_time += 1
        if embeddings:
            return embeddings
        if request_time>=request_time_patience:
            model = args["model"]
            print(f"{model} request failed!")
            return None
        
def ensure_directory_exists(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def pretty_print_dict(input_dict):
    pretty_str = json.dumps(input_dict, indent=4, sort_keys=True)
    print(pretty_str)

def add_suffix_to_filename(filename, suffix="_tmp"):
    name, ext = os.path.splitext(filename)
    new_filename = f"{name}{suffix}{ext}"
    return new_filename


def get_subtrees(tree_node, search_value_list):
    def get_subtree(tree_node, search_value):
        if search_value in tree_node.value and tree_node.node_type == "Intermediate Node":
            return tree_node
        for child_tree_node in tree_node.children.values():
            result = get_subtree(child_tree_node,search_value)
            if result:
                return result
        return None
    subtrees = [get_subtree(tree_node, search_value) for search_value in search_value_list]
    
    # Delete duplicated or parent-child trees

    return subtrees

