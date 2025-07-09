import os
import time
import json

from utils import *
from glob import glob
from tqdm import tqdm
from bs4 import BeautifulSoup
from functools import partial
from concurrent.futures import ProcessPoolExecutor, as_completed

def list_to_tree(data_list):
    root = TreeNode(data_list[0]["h1"],"h1")
    for item in data_list:
        current_node = root
        for key in ["h2", "h3", "h4"]:
            value = item.get(key)
            if value is not None:
                child_node = TreeNode(value,key)
                current_node = current_node.add_child(child_node)
        leaf_node = TreeNode(item["text"],"leaf")
        current_node.add_child(leaf_node)
    return root

def extract_content(soup:BeautifulSoup):
    def no_attrs(tag):
        return tag.name in ["h2", "h3", "h4", "p", "div", "li"] and len(tag.attrs) == 0
    
    h1 = soup.find("h1").get_text().strip()
    content = soup.find("div",attrs={"id":"bodyContentOuter"})
    tb_tags = content.find_all('tbody')
    for tb_tag in tb_tags:
        tb_tag.decompose()
    content = content.find_all(no_attrs)
    
    h2 = h3 = h4 =  None
    content_list = []
    for tag in content:
        tag_name = tag.name
        tag_text = tag.get_text()
        if (tag_name=="h1" or tag_name=="h2") and (tag_text=="参看" or tag_text=="参考"):
            break
        if tag_name=="h2":
            h2 = tag_text
            h3 = None
            h4 = None
        elif tag_name=="h3":
            h3 = tag_text
            h4 = None
        elif tag_name=="h4":
            h4 = tag_text
        else:
            tag_text = tag_text.strip()
            if tag_text:
                content_list.append({"text":tag_text,"h1":h1,"h2":h2,"h3":h3,"h4":h4})
    
    return content_list

def process_item(item):
    tree_root_dict = dict()
    try:
        with open(item,mode="r",encoding="utf-8") as file:
            soup = BeautifulSoup(file.read(), 'html.parser')
            content_list = extract_content(soup=soup)
            tree_root = list_to_tree(content_list)
            name = tree_root.value
            tree_root_dict = tree_root.to_dict()
    except Exception as e:
        print(f"Error happens when processing {item}:\n{e}")
    return tree_root_dict


def process_items_with_pool(items):
    results = []
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_item, item) for item in items]
        for future in tqdm(as_completed(futures), total=len(items), desc="Processing items"):
            result = future.result()
            if result:
                results.append(result)
    return results

if __name__=="__main__":
    script_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_path)

    html_dir = os.path.join(script_dir,r"htmls")
    save_path = os.path.join(script_dir,r"diseases.json")

    file_paths = glob(html_dir + "/*.html")

    # results = process_items_with_pool(file_paths)

    results = []
    for file_path in tqdm(file_paths):
        result = process_item(file_path)
        results.append(result)
    
    with open(save_path,mode="w",encoding="utf-8") as file:
        file.write(json.dumps(results, indent=4, ensure_ascii=False))
