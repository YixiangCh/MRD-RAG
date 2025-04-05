#coding=utf-8

import os
import json
import argparse

from utils import *
from tqdm import tqdm
from MRD_RAG import *

if __name__=="__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--neo4j_uri',type=str,default=None)
    parser.add_argument('--neo4j_username',type=str,default=None)
    parser.add_argument('--neo4j_password',type=str,default=None)
    parser.add_argument('--neo4j_database',type=str,default=None)

    parser.add_argument('--index_name',type=str,default=None,choices=["diagnosis info", "pseudo medical history"])
    parser.add_argument('--need_desc',type=bool,default=True)

    parser.add_argument('--embedding_model',type=str,default=None)
    parser.add_argument('--embedding_api_base',type=str,default=None)
    parser.add_argument('--embedding_api_key',type=str,default=None)
    
    parser.add_argument('--patient_api_base',type=str)
    parser.add_argument('--patient_api_key',type=str)
    parser.add_argument('--patient_model',type=str)
    parser.add_argument('--patient_model_max_tokens',type=int,default=3072)

    parser.add_argument('--doctor_api_base',type=str)
    parser.add_argument('--doctor_api_key',type=str)
    parser.add_argument('--doctor_model',type=str)
    parser.add_argument('--doctor_model_max_tokens',type=int,default=3072)

    parser.add_argument('--test_data_file_path',type=str)
    parser.add_argument('--doctor_type',type=str,choices=["TCM","MM"])

    parser.add_argument('--max_round',type=int)

    parser.add_argument('--topk',type=int,default=5)
    parser.add_argument('--disease_info_max_tokens',type=int,default=15000)
    
    parser.add_argument('--resume',type=str2bool,default=False)
    parser.add_argument('--save_dir',type=str)
    parser.add_argument('--save_name_tmp',type=str,default=None)
    parser.add_argument('--save_name',type=str)

    args = parser.parse_args()
    pretty_print_dict(vars(args))
    print("======================================================================================")

    solver = MRD_RAG(args=args)

    patient_llm_args = {
    "openai_api_base": args.patient_api_base,
    "openai_api_key": args.patient_api_key,
    "model": args.patient_model,
    "temperature": 0,
    "max_tokens": args.patient_model_max_tokens,
    "presence_penalty": 0,
    }
    
    ensure_directory_exists(args.save_dir)
    save_path = os.path.join(args.save_dir,args.save_name)
    if not args.save_name_tmp:
        save_name_tmp = add_suffix_to_filename(args.save_name,"_tmp")
        save_path_tmp = os.path.join(args.save_dir,save_name_tmp)

    log_step = 1

    patient_infos = get_patient_infos(args.test_data_file_path)

    start_id = 0
    results_dict = dict()
    if args.resume:
        with open(save_path_tmp,mode="r",encoding="utf-8") as file:
            resume_results = json.load(file)
            for i,result in enumerate(resume_results):
                if "error" not in result["state"]:
                    results_dict[result["patient info"]] = result
            print(f"finished: {len(results_dict)}, remaining: {len(patient_infos)-len(results_dict)}...")
    
    results = []
    for i,patient_info in tqdm(enumerate(patient_infos,start=start_id)):
        patient_info_str = patient_info["patient info"]
        if patient_info_str in results_dict:
            result = results_dict[patient_info_str]
        else:
            disease_name = patient_info["disease name"]
            history,reasoning_process,state = solver.diagnose(patient_info=patient_info,patient_llm_args=patient_llm_args)
            result = patient_info
            result["state"] = state
            result["dialog history"] = history
            result["doctor LLM reasoning process"] = reasoning_process
        results.append(result)
        if i%log_step == 0:
            with open(save_path_tmp,mode="w",encoding="utf-8") as file:
                file.write(json.dumps(results, indent=4, ensure_ascii=False))


    with open(save_path,mode="w",encoding="utf-8") as file:
        file.write(json.dumps(results, indent=4, ensure_ascii=False))

    