import os
import json
import random
import pandas as pd
from tqdm import tqdm
from datetime import datetime

from Instructions import Instructions
from utils import *

def main(start_idx=300, end_idx=400, save_every=100, seed=0):

    random.seed(seed)

    inst_type_weights = {
        InstType.ADD_OBS: 8,
        InstType.RM_OBS: 3,
        InstType.CHG_OBS: 3,
        InstType.CHG_LOC_OF_OBS: 1,
        InstType.CHG_SHAPE_OF_OBS: 1,
        InstType.CHG_SEVERITY: 1,
        InstType.CHG_CERTAINTY: 1,
        InstType.ADD_REC: 1,
        InstType.RM_REC: 1,
        InstType.CHG_REC: 1,
        InstType.ADD_COMPS_TO_PRIOR: 1,
        InstType.RM_COMPS_TO_PRIOR: 1,
        InstType.CHG_COMPS_TO_PRIOR: 1
    }

    location_weights = {
        Location.REPORT: 4,
        Location.SECTION: 2,
        Location.SECTION_FIND:2,
        Location.SECTION_IMPR:2,
        Location.LINE: 4,
        Location.LINE_FIRST:0,
        Location.LINE_LAST:0
    }

    inst_maker = Instructions(inst_type_weights=inst_type_weights, location_weights=location_weights)    
    all_inst = inst_maker.get_all_inst()
    res = {
        'inst_type': [],
        'location': [],
        'topic': []
    }

    for inst_type in all_inst.keys():
        locations = all_inst[inst_type]
        for location in locations.keys():
            topics = locations[location]
            for topic in topics:
                res['inst_type'].append(inst_type.value)
                res['location'].append(location.value)
                res['topic'].append(topic.value)
    res = pd.DataFrame(res)
    res.to_csv('../output/all_inst.csv', index=False)

    with open('../data/test.json', 'r') as json_file:
        reports = json.load(json_file)

    print(f"len is {end_idx-start_idx}")

    data = []
    total_cost = 0

    now = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    os.makedirs(f'../output/output_{now}', exist_ok=True)

    reports = reports[start_idx:end_idx]
    for i, report in tqdm(enumerate(reports), total = len(reports)):

        n_inst = random.choices([1, 2, 3, 4, 5], weights=[4, 2, 2, 1, 1])[0]
        user_inst = inst_maker.get_insts(report['report_text'], n=n_inst)

        if user_inst is None:
            continue

        response, cost = query_openai(user_inst, report['report_text'])

        data.append({
            'original': report,
            'n_inst': n_inst,
            'user_inst': user_inst, 
            'response': response
        })

        total_cost += cost
        if ((i+1)%save_every==0):
            print(f"Total cost by {i} report is {total_cost}")
            with open(f'../output/output_{now}/records_{i}.json', 'w') as f:
                json.dump(data, f, indent=4)

            data = []

if __name__=='__main__':
    main()