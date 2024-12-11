from Core.GraphRAG import GraphRAG
from Option.Config2 import Config
import argparse
import os
import asyncio
from pathlib import Path
from shutil import copyfile

parser = argparse.ArgumentParser()
parser.add_argument('-opt', type=str, help='Path to option YMAL file.')
args = parser.parse_args()

opt = Config.parse(Path(args.opt))
digimon = GraphRAG(config  = opt)

def check_dirs(opt):
    result_dir = os.path.join(opt.working_dir,opt.exp_name, "Results") # For each query, save the results in a separate directory
    config_dir = os.path.join(opt.working_dir,opt.exp_name,  "Configs") # Save the current used config in a separate directory
    metric_dir = os.path.join(opt.working_dir,  opt.exp_name, "Metrics") # Save the metrics of entire experiment in a separate directory
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(config_dir, exist_ok=True)
    os.makedirs(metric_dir, exist_ok=True)
    opt_name = args.opt[args.opt.rindex('/') + 1:]
    basic_name = os.path.join(os.path.dirname(args.opt), "Config2.yaml")
    copyfile(args.opt, os.path.join(config_dir, opt_name))
    copyfile(basic_name, os.path.join(config_dir, "Config2.yaml"))

if __name__ == "__main__":
    check_dirs(opt)
    with open("./book.txt") as f:
        doc = f.read()
    asyncio.run(digimon.insert([doc]))
    
    asyncio.run(digimon.query("Who is Scrooge?"))
