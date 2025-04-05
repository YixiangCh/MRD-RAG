
# MRD-RAG



<div style="text-align: center;">
  <img src="assets\MRD_RAG_Pipeline.jpeg" alt="image description" width="400" />
</div>

<!-- ## 项目介绍 -->

This repo provides the source code & data of our paper: "MRD-RAG: Mimicking Clinical Diagnosis Through Multi-round Medical Dialogue Analysis".

## How to run

To run this project, follow these steps:

1. Import our DI-Tree KB into neo4j. The dump files are availeble <a href="[DI-Tree KB](https://pan.baidu.com/s/1I3A7k-Yac5ecB2X5YZde8w?pwd=vwe5)">here</a>. You can use the following command to dump our KB into your own neo4j database.

   ```bash
   neo4j-admin load --from="DI-Tree KB/modern-medicine-disease.dump" --database="NAME_OF_YOUR_DATABASE" --force
   ```

2. Navigate to the project directory：

   ```bash
   cd MRD-RAG
   ```

3. Set your openai key and other necessary parameters in the script file. Then you can run the project and get the diagnosis result：

   ```bash
   sh scripts\CMB\CMB_MRD-RAG-PMH_gpt-4o-mini.sh
   ```


