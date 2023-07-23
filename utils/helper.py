
import json
# This are some functions to help in loading and saving json file

def save_json(data,name):
    with open(name+'.json', 'w',encoding='utf8') as f:
        json.dump(data, f,ensure_ascii=False)
        
def save_jsonl(data,name):
    with open(name+'.jsonl', 'w') as f:
        for entry in data:
            json.dump(entry, f)
            outfile.write('\n')

def load_json(path):
    f = open (path, "r")
    data = json.loads(f.read())
    f.close()
    return data

def load_jsonl(path):
    f = open(path,"r").readlines()
    data=[]
    for i in f:
        d = json.loads(i)
        data.append(d)
    return data