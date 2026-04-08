import json
output_path="/fs-computility/MA4Tool/yuzhiyin/RLEvol/save/tmp/mathscale__fs-computility_MA4Tool_shared_MA4Tool_hug_ckpts_Qwen2.5-Math-1.5B_0717_054857.json"
with open(output_path, 'r', encoding='utf-8') as f:
    for line in f:
        print(json.loads(line)['response'])
