import json
import os
import sys
import pandas as pd

def main():
    eval_path = sys.argv[1]
    with open(os.path.join(eval_path, 'results.json'), 'rb') as f:
        data = f.read().decode('ascii')
    data = data.replace(',\n}','\n}')
    json_data = json.loads(data)
    df = pd.DataFrame(json_data)
    for eval_type in ['document', 'paragrph']:
         df.loc[df['eval_type'] == f'{eval_type}'].drop('eval_type', 1).to_csv(os.path.join(eval_path, f'{eval_type}_eval.csv'))


if __name__ == '__main__':
    main()
