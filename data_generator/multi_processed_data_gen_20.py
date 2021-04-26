import argparse
import numpy as np
import pandas as pd
import os
from data_gen_20 import uni_data_generator
from multiprocessing import Pool, Manager

# parsing user input option
parser = argparse.ArgumentParser(description='Data generator')
parser.add_argument('--num_workers', type=int,
                    default=60, help='num workers')
parser.add_argument('--num_data', type=int,
                    default=300000, help='num data')
parser.add_argument('--noised', type=lambda s: s.lower in ["true", 1],
                    default=True, help='noise for data')
args = parser.parse_args()


def work(number_of_loops, result_queue, noised):
    work_result = uni_data_generator(number_of_loops, True, noised)
    result_queue.put(work_result)
    return


def multiprocess_data_gen(num_workers_in, number_of_data_in, noised):
    number_of_works = int(number_of_data_in/num_workers_in)
    pool = Pool(num_workers_in)
    m = Manager()
    result_queue = m.Queue()
    pool.starmap(work, [(number_of_works, result_queue, noised) for _ in range(num_workers_in)])
    pool.close()
    pool.join()
    return result_queue


if __name__ == "__main__":
    num_workers = args.num_workers
    number_of_data = args.num_data
    noise = args.noised
    if noise:
        file_name = "_noised"
    else:
        file_name = ""
    if os.path.exists("kf_train"+file_name+"_20.csv") is False:
        # train data
        data = multiprocess_data_gen(num_workers, number_of_data, noise)
        data.put("stop")
        res = []
        while True:
            data_piece = data.get()
            if data_piece == "stop":
                break
            else:
                res = res + data_piece
        res = np.array(res)
        df = pd.DataFrame(res)
        df.to_csv("kf_train"+file_name+"_20.csv", header=False, index=False)
        print("Train data generation complete")
    if os.path.exists("kf_val"+file_name+"_20.csv") is False:
        # validation data
        data = multiprocess_data_gen(num_workers, int(number_of_data/10), noise)
        data.put("stop")
        res = []
        while True:
            data_piece = data.get()
            if data_piece == "stop":
                break
            else:
                res = res + data_piece
        res = np.array(res)
        df = pd.DataFrame(res)
        df.to_csv("kf_val"+file_name+"_20.csv", header=False, index=False)
        print("validation data generation complete")
