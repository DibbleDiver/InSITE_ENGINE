import pandas as pd
import glob
import os

def load_folder(path):
    if not os.path.exists(path):
        return []
    files = glob.glob(os.path.join(path, '*.csv'))
    return [pd.read_csv(f) for f in files]

def run_ingestion(cfg):
    data = {}
    base = cfg['input_path']
    for src in ['TRIO','TCL','AX','Meters']:
        folder = os.path.join(base, src)
        data[src] = load_folder(folder)
    return data
