from datasets import load_dataset
from pathlib import Path
import json

out = Path('/root/autodl-tmp/data/pokerbench_train.jsonl')
meg = Path('/root/autodl-tmp/data/pokerbench_megatron_debug.jsonl')
smk = Path('/root/autodl-tmp/data/pokerbench_holdem_smoke.jsonl')
out.parent.mkdir(parents=True, exist_ok=True)

ds = load_dataset('RZ412/PokerBench', split='train')
keys = set(ds.column_names)
need = {'instruction', 'output'}
if not need.issubset(keys):
    raise RuntimeError(f'missing required keys: {need-keys}; all keys={keys}')

ds.to_json(str(out), force_ascii=False)

def write_subset(src, dst, n):
    c = 0
    with open(src, 'r', encoding='utf-8') as fin, open(dst, 'w', encoding='utf-8') as fout:
        for line in fin:
            if not line.strip():
                continue
            obj = json.loads(line)
            if 'instruction' not in obj or 'output' not in obj:
                continue
            fout.write(json.dumps({'instruction': obj['instruction'], 'output': obj['output']}, ensure_ascii=False) + '\n')
            c += 1
            if c >= n:
                break
    if c == 0:
        raise RuntimeError(f'no valid samples written to {dst}')

write_subset(out, meg, 256)
write_subset(out, smk, 64)
print('DONE', out, meg, smk)
