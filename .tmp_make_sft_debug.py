import json
from pathlib import Path
src = Path('/root/autodl-tmp/data/pokerbench_megatron_debug.jsonl')
dst = Path('/root/autodl-tmp/data/pokerbench_megatron_sft_messages.jsonl')
count = 0
with src.open('r', encoding='utf-8') as fin, dst.open('w', encoding='utf-8') as fout:
    for line in fin:
        if not line.strip():
            continue
        obj = json.loads(line)
        ins = obj.get('instruction')
        out = obj.get('output')
        if not isinstance(ins, str) or not isinstance(out, str):
            continue
        rec = {
            'messages': [
                {'role': 'user', 'content': ins},
                {'role': 'assistant', 'content': out},
            ]
        }
        fout.write(json.dumps(rec, ensure_ascii=False) + '\n')
        count += 1
print('written', count, dst)
