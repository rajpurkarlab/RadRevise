import argparse
import pandas as pd
from CXRMetric.run_eval import calc_metric
from inference import *

def main(args):
    data = pd.read_csv(args.data_path)
    gt_reports, predicted_reports = postprocess(run_inference(args.model_id, data, args.batch_size))
    calc_metric(gt_reports, predicted_reports, args.out_file, False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('model_id', type=str, help="Model ID on HF")
    parser.add_argument('data_path', type=str, default='../data/RadRevise_v0.csv', help="RadRevise dataset")
    parser.add_argument('inference_batch_size', default=32, type=int)
    parser.add_argument('out_file', type=str, default='../output/result', help="RadRevise dataset")
    args = parser.parse_args()
    main(args)
