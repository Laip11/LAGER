import json
import re
two_digit_data =[]
with open('flask.json', 'r',encoding='utf-8') as f:
    for line in f:
        new_data = {}
        data = json.loads(line)

        prompt = data['user_prompt']
        pattern = r'### Score Rubrics:\n(.*?)\n### Output format:'
        match = re.search(pattern, prompt, re.DOTALL)
        result = match.group(1)  # 提取匹配的内容
        del_str = result
        for num in [5,9,19,29,39,49,59,69,79,89,99]:
            new_prompt = prompt.replace(del_str,'').replace('an integer between 1 and 5',f'an integer between 1 and {num}').replace('strictly based on the given score rubric','').replace('### Score Rubrics:\n\n','')
            new_data[f'score_{num}_prompt'] = new_prompt
        two_digit_data.append(new_data)
with open('two_digit_data.json','w',encoding='utf-8') as f:
    json.dump(two_digit_data,f,ensure_ascii=False,indent=4)
