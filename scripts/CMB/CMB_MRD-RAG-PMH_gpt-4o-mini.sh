python ./MRD-RAG/main.py \
--neo4j_uri "bolt://localhost:7687" --neo4j_username neo4j --neo4j_password 12345678 --neo4j_database "modern-medicine-disease" \
--index_name "pseudo medical history" --embedding_model "text-embedding-3-small" --embedding_api_base "embedding_api_base" --embedding_api_key "embedding_api_key" \
--patient_api_base "patient_api_base" --patient_api_key "patient_api_key" --patient_model "gpt-4o-mini" \
--doctor_api_base "doctor_api_base" --doctor_api_key "doctor_api_key" --doctor_model "gpt-4o-mini" \
--disease_info_max_tokens 11000 --max_round 3 --topk 5 \
--test_data_file_path "./data/CMB.json" --doctor_type "MM" --save_dir "./results/CMB"  --save_name "CMB_MRD-RAG-PMH_gpt-4o-mini.json"
