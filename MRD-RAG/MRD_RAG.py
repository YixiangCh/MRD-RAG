#coding=utf-8

import math

from utils import *
from collections import defaultdict
from langchain_community.graphs import Neo4jGraph
from langchain_text_splitters import RecursiveCharacterTextSplitter,CharacterTextSplitter

class MRD_RAG:
    def __init__(self,args):
        self.kg = Neo4jGraph(url=args.neo4j_uri, username=args.neo4j_username, password=args.neo4j_password, database=args.neo4j_database)
        self.index_name = args.index_name
        self.need_desc = args.need_desc
        self.doctor_type = args.doctor_type
        if self.doctor_type=="TCM":
            self.search_list = ["症状","病因","表现"]
        elif self.doctor_type=="MM":
            self.search_list = ["表现","症状","诊断","检查"]

        self.embedding_args = {
        "openai_api_base": args.embedding_api_base,
        "openai_api_key": args.embedding_api_key,
        "model": args.embedding_model,
        }

        self.doctor_llm_args = {
        "openai_api_base":  args.doctor_api_base,
        "openai_api_key": args.doctor_api_key,
        "model": args.doctor_model,
        "temperature": 0,
        "max_tokens": args.doctor_model_max_tokens,
        "presence_penalty": 0,
        }

        separators = ["\n",".","．","。","?","？","!","！",";","；",":","：",",","，","、"," ","\u200B",""]
        self.splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            model_name="gpt-3.5-turbo",
            separators=separators,
            chunk_size=4096,
            chunk_overlap=0
            )

        self.multi_round_args = {
            "max_round": args.max_round,
            "retrieval_num": 20,
            "topk": args.topk,
            "disease_info_max_tokens": args.disease_info_max_tokens,  # The sum of the maximum text lengths for all diseases
        }

    def query_embedding(self,kg,index_name,query,embedding_args,result_num=10):
        embedding = get_embeddings_retry(inputs=[query],args=embedding_args)[0]
        if embedding:
            query_embedding_cypher = """CALL db.index.vector.queryNodes(\"%s\", %d, $embedding)\nyield score, node\nRETURN score, node, elementid(node) AS node_id"""%(index_name,result_num)
            embedding_query_results = kg.query(query_embedding_cypher,params={"embedding":embedding})
            return embedding_query_results
        else:
            return None

    def get_candidate_diseases(self,kg,embedding_query_results,topk,splitter,search_list=[],need_desc=False):
        candidate_diagnosis_information_list = []
        candidate_disease_name_list = []
        for embedding_query_result in embedding_query_results:
            if len(candidate_disease_name_list)>=topk:
                break        
            score, node, node_id = embedding_query_result["score"],embedding_query_result["node"],embedding_query_result["node_id"]
            source = node["source"]
            disease_name = node["name"]     
            if disease_name in candidate_disease_name_list:
                continue

            embedding_node2first_tree_node_cypher = """MATCH (embedding_node)<-[*]-(main_node)-[]->(info_tree_root:`Info Tree Root`{source: "%s"})-[]->(first_info_tree_node)\nWHERE elementid(embedding_node)="%s"\nRETURN elementid(first_info_tree_node) as node_id"""%(source,node_id)
            root_node_uuid=kg.query(embedding_node2first_tree_node_cypher)[0]["node_id"]
            DI_tree = restore_info_tree(kg=kg,root_node_uuid=root_node_uuid)
            diagnosis_information = get_info(DI_tree,splitter,search_list=search_list,need_desc=need_desc)
            candidate_disease_name_list.append(disease_name)
            candidate_diagnosis_information_list.append(diagnosis_information)
        return candidate_disease_name_list,candidate_diagnosis_information_list

    def adjustTexts(self,text_list, threshold, model_name): # Adjust the length of the text in the array so that the total length of the text is limited to the threshold, while ensuring that the adjusted length of each text keeps the length order as far as possible, and the short priority to the maximum length. The order of the text returned may be different from the original order
        length_list = [get_num_tokens(text,model_name=model_name) for text in text_list] 
        total_length = sum(length_list)
        if total_length <= threshold:
            return text_list
        
        ratio = threshold / total_length
        text_list_new = []
        allocated_length_init_list = []
        
        text_and_len_sorted = sorted(enumerate(list(zip(text_list,length_list))), key=lambda x: x[1][1])  
        original_indexs = [item[0] for item in text_and_len_sorted]

        
        for original_index,(text,length) in text_and_len_sorted:
            allocated_length = math.floor(length * ratio)
            separators = ["\n",".","．","。","?","？","!","！",";","；",":","：",",","，","、"," ","\u200B",""]
            splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(model_name=model_name, separators=separators, chunk_size=allocated_length,chunk_overlap=0)        
            text_init = splitter.split_text(text=text)[0]

            allocated_length_init_real = get_num_tokens(text_init,model_name=model_name)
            allocated_length_init_list.append(allocated_length_init_real)
            text_list_new.append(text_init)
        
        remaining = threshold - sum(allocated_length_init_list)

        for i in range(len(text_and_len_sorted)):
            if remaining > 0:
                allocated_length = allocated_length_init_list[i] + remaining
                splitter_chr = RecursiveCharacterTextSplitter.from_tiktoken_encoder(model_name=model_name, separators=[""], chunk_size=allocated_length,chunk_overlap=0)   
                text_new = splitter_chr.split_text(text=text_and_len_sorted[i][1][0])[0]
                text_list_new[i] = text_new
                allocated_length_real = get_num_tokens(text_new,model_name=model_name)
                remaining = remaining + allocated_length_init_list[i] - allocated_length_real
            else:
                break

        text_list_new = sorted(zip(original_indexs,text_list_new),key=lambda x: x[0])
        text_list_new = [item[1] for item in text_list_new]
        return text_list_new

    def organize_candidate_diagnosis_information_str(self,candidates, disease_info_max_tokens, model_name):

        candidate_disease_name_list,candidate_diagnosis_information_list = zip(*candidates)
        
        candidate_diagnosis_information_list = self.adjustTexts(text_list=candidate_diagnosis_information_list, threshold=disease_info_max_tokens, model_name=model_name)
        candidate_diagnosis_information_str = ""
        for disease_name,diagnosis_info in zip(candidate_disease_name_list,candidate_diagnosis_information_list):
            candidate_diagnosis_information_str = candidate_diagnosis_information_str + f"""以下为关于“{disease_name}”的基本信息：\n{diagnosis_info}\n\n"""

        return candidate_diagnosis_information_str 

    def retrieve_diseases(self,kg,history,embedding_args,multi_round_args):
        query_list = []
        for utterance in history:
            if utterance["role"]=="patient":
                query_list.append(utterance["content"])
            else:
                continue
        query_str = "\n".join(query_list)
        embedding_query_results = self.query_embedding(kg=kg,index_name=self.index_name,query=query_str,result_num=multi_round_args["retrieval_num"],embedding_args=embedding_args)
        if not embedding_query_results:
            return None
        candidate_disease_name_list,candidate_diagnosis_information_list = self.get_candidate_diseases(kg=kg,embedding_query_results=embedding_query_results,topk=multi_round_args["topk"],splitter=self.splitter,search_list=self.search_list,need_desc=self.need_desc) 
        candidates = list(zip(candidate_disease_name_list,candidate_diagnosis_information_list))
        candidate_diagnosis_information_str = self.organize_candidate_diagnosis_information_str(candidates=candidates,disease_info_max_tokens=multi_round_args["disease_info_max_tokens"],model_name="gpt-3.5-turbo")
        return candidate_disease_name_list,candidate_diagnosis_information_str

    def get_K_thinking_prompt(self,history, candidate_disease_infos_str,doctor_type):
        doctor_role = "中医" if doctor_type=="TCM" else "医生"
        disease_type = "症候" if doctor_type=="TCM" else "疾病"

        if doctor_type=="TCM":
            sys_content = f"""你是一个AI中医。接下来我将提供一些症候的基本信息和一段医疗问诊对话给你。请你按照要求完成以下任务，任务的具体的要求如下：\n1.首先，请你逐个总结各个症候的症状、病因，临床表现、望闻问切四诊信息等，把不同症候归类，分析各症候之间的区别与联系，并总结如何辨别诊断这些症候。\n2.然后，逐个分析患者与各个症候之间的联系（若多个症状与该症候相符合，则患者可能患有该症候，需要指出这些症状；若没有症状与该症候符合，则患者可能并非患有该症候）；此外，可能还需要考虑患者的年龄、性别等其他因素；需要注意的是，患者也可能患有所提供的症候以外的其他症候，如果患者所患有的症状与所提供的症候全都不符合，也请指出。\n3.进一步地，思考目前已知患者的哪些医疗信息，还缺失哪些关键信息（例如患者性别，年龄，是否存在某些症状，望闻问切四诊信息等），思考应该向患者提问问题还是指出患者可能患有的症候。请给出思考的过程。"""   
        else:
            sys_content = f"""你是一个AI医生。接下来我将提供一些疾病的基本信息和一段医疗问诊对话给你。请你按照要求完成以下任务，任务的具体的要求如下：\n1.首先，请你逐个总结各个疾病的症状、病因、临床表现、诊断检查等，把不同疾病归类，分析各疾病之间的区别与联系，并总结如何辨别诊断这些疾病。\n2.然后，逐个分析患者与各个疾病之间的联系（若多个症状或检查结果与该疾病相符合，则患者可能患有该疾病，需要指出这些症状或检查结果；若没有症状或检查结果与该疾病符合，则患者可能并非患有该疾病）；此外，可能还需要考虑患者的年龄、性别等其他因素；需要注意的是，患者也可能患有所提供的疾病以外的其他疾病，如果患者所患有的症状与所提供的疾病全都不符合，也请指出。\n3.进一步地，思考目前已知患者的哪些医疗信息，还缺失哪些关键信息（例如患者性别，年龄，是否存在某些症状，做过哪些必要的医学检查，结果如何等），思考应该向患者提问问题还是指出患者可能患有的疾病。请给出思考的过程。"""

        history_str = "以下为一个医疗问诊对话：\n[对话开始]\n"
        for i,utterance in enumerate(history):
            role = f"{doctor_role}" if utterance["role"]=="doctor" else "患者"
            history_str += role + "：" + utterance["content"] + "\n"
        history_str = history_str + "[对话结束]\n"

        usr_content = f"""以下为一些{disease_type}的基本信息：\n""" + candidate_disease_infos_str + "\n" + history_str + f"""请按照要求逐步进行总结，思考和分析：\n"""
        prompt = [
            {"role": "system", "content": sys_content},
            {"role": "user", "content": usr_content}
        ]
        return prompt

    def get_doctor_prompt(self,history, K_thinking):
        doctor_role = "中医" if self.doctor_type=="TCM" else "医生"
        history_str = "以下为一个医疗问诊对话：\n"
        for i,utterance in enumerate(history):
            role = f"{doctor_role}" if utterance["role"]=="doctor" else "患者"
            history_str += role + "：" + utterance["content"] + "\n"
        history_str = history_str + f"{doctor_role}："

        if K_thinking:
            if self.doctor_type=="TCM":
                sys_content = """你是一个AI中医，正在与一名患者进行对话并尝试对患者进行“辨证”。关于患者的“症候”，你已经有了一些“思考”。请你根据“思考”和对话的内容，对患者进行回复。作为中医，你回复的语气应当友好，符合中医的说话风格。\n""" 
            else:
                sys_content = """你是一个AI医生，正在与一名患者进行对话并尝试对患者进行诊断。关于患者的疾病，你已经有了一些“思考”。请你根据“思考”和对话的内容，对患者进行回复。作为医生，你回复的语气应当友好，符合医生的说话风格。\n"""

            K_thinking = """以下为你此前的思考过程：\n[思考开始]\n""" + K_thinking + """\n[思考结束]\n"""
            usr_content = K_thinking + history_str
        else:
            if self.doctor_type=="TCM":
                sys_content = """你是一个AI中医，正在与一名患者进行对话并尝试对患者进行“辨证”。请你根据之前对话的内容，对患者进行回复。作为中医，你回复的语气应当友好，符合中医的说话风格。\n""" 
            else:
                sys_content = """你是一个AI医生，正在与一名患者进行对话并尝试对患者进行诊断。请你根据之前对话的内容，对患者进行回复。作为医生，你回复的语气应当友好，符合医生的说话风格。\n"""
            
            usr_content = history_str
        prompt = [
            {"role": "system", "content": sys_content},
            {"role": "user", "content": usr_content}
        ]

        return prompt

    def get_need_retrieval_prompt(self,history):
        sys_content = f"""接下来，我将为你提供一段医疗问诊对话，请你判断在对话中，患者所说的“最新一句话”是否相比其之前所说的话提供了更多有利于对其诊断的医疗信息。例如：患者的最新一句话包含了自身的更多症状、医学检查结果等，则回复字符串"True"；若患者的最新一句话提出了一个不含有自身医疗信息的问题或者没有描述自身医疗信息等，例如仅仅表示感谢医生等，则回复字符串"False"。在输出"True"或者"False"之前，请先进行逐步推理。"""   

        history_str = "以下为一个医疗问诊对话：\n[对话开始]\n"
        for i,utterance in enumerate(history):
            role = f"医生" if utterance["role"]=="doctor" else "患者"
            if i==len(history)-1:
                history_str += role + "（最新一句话）：" + utterance["content"] + "\n"
            else:
                history_str += role + "：" + utterance["content"] + "\n"
        history_str = history_str + "[对话结束]\n"

        usr_content = history_str + f"""请按照要求先逐步推理，再输出"True"或者"False"：\n"""
        prompt = [
            {"role": "system", "content": sys_content},
            {"role": "user", "content": usr_content}
        ]
        return prompt    


    def need_retrieval(self,history):
        need_retrieval_prompt = self.get_need_retrieval_prompt(history)
        need_retrieval_response = run_llm_retry(prompt=need_retrieval_prompt,llm_args=self.doctor_llm_args)
        if "True" in need_retrieval_response:
            return True
        else:
            return False

    def Retriever(self,history):
        candidate_disease_name_list,candidate_diagnosis_information_str = self.retrieve_diseases(self.kg,history,self.embedding_args,self.multi_round_args)
        if not candidate_diagnosis_information_str:
            return None,None
        return candidate_disease_name_list,candidate_diagnosis_information_str        

    def Analyzer(self,candidate_disease_infos_str,history):
        thinking_prompt = self.get_K_thinking_prompt(candidate_disease_infos_str=candidate_disease_infos_str,history=history,doctor_type=self.doctor_type)
        K_thinking = run_llm_retry(prompt=thinking_prompt,llm_args=self.doctor_llm_args)
        if not K_thinking:
            return None
        K_thinking = K_thinking + "\n"
        return K_thinking

    def Doctor(self,history,K_thinking):
        doctor_prompt = self.get_doctor_prompt(history=history,K_thinking=K_thinking)
        doctor_llm_response = run_llm_retry(prompt=doctor_prompt,llm_args=self.doctor_llm_args)
        return doctor_llm_response

    def diagnose(self,patient_info, patient_llm_args):
        history = [{"role":"doctor","content":"您好，我是一名医生，请问有什么可以帮您？"}]
        reasoning_process = defaultdict(list)
        
        for round in range(self.multi_round_args["max_round"]):
            patient_llm_response = Patient(round,patient_info,history,patient_llm_args)
            if not patient_llm_response:
                return history,reasoning_process,"patient api error" 
            
            history.append({"role":"patient", "content":patient_llm_response})

            
            candidate_disease_name_list = None
            K_thinking = None
            if round==0:
                retrieval = True
            else:
                retrieval = self.need_retrieval(history)
            if retrieval:
                candidate_disease_name_list,candidate_diagnosis_information_str = self.Retriever(history)
                if not candidate_diagnosis_information_str:
                    return history,reasoning_process,"embedding api error"
                        
                K_thinking = self.Analyzer(candidate_diagnosis_information_str,history)
                if not K_thinking:
                    return history,reasoning_process,"analyzer api error"
                    
            reasoning_process["candidate diseases"].append(candidate_disease_name_list)      
            reasoning_process["K_thinking"].append(K_thinking)

            doctor_llm_response = self.Doctor(history,K_thinking)
            if not doctor_llm_response:
                return history,reasoning_process,"doctor api error"   
            history.append({"role":"doctor", "content":doctor_llm_response})

        return history,reasoning_process,"exceed maximum round" # return normally
