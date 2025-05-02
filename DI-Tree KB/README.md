# Build DI-Tree KB
The original DI-Tree KB dump files used in our experiments are availeble <a href="https://drive.google.com/drive/folders/1IhO7lBV674Jk95q71tJUMNZDRca8LFPt?usp=sharing">here</a>. If you want to build your own KB, you can read this README file and go ahead.

To build your own DI-Tree KB, follow these steps:

1. The structure of different medical encyclopedia web pages varies. So first you need to collect the pages and parse them into a unified JSON format. Here, we provide a processing script, `parse_yixue_diseases_htmls.py`, for parsing disease HTML pages from the <a href="https://www.yixue.com/">yixue</a> website. You can use it directly or modify it according to the structure of the pages you need to parse. Run `parse_yixue_diseases_htmls.py` and you'll get the `disease.json` file.

2. Create a database manually and name it (e.g. `modern-medicine-disease`). Besides, APOC plugin is required for your neo4j database.

3. Modify the neo4j password and other parameters in the `store2neo4j.py` file and run it to store the above JSON file into the neo4j DB.

4. Modify and run the `indexing.py` file (you should set your own openai api key) for indexing.