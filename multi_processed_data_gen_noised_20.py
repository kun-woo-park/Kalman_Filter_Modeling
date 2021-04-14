import argparse
import numpy as np
import pandas as pd
import os
from data_gen_noised import uni_data_generator
from multiprocessing import Pool, Manager

# parsing user input option
parser = argparse.ArgumentParser(description='Data generator')
parser.add_argument('--num_workers', type=int,
                    default=60, help='num workers')
parser.add_argument('--num_data', type=int,
                    default=300000, help='num data')
args = parser.parse_args()


def work(number_of_loops, result_queue):
    work_result = uni_data_generator(number_of_loops, True)
    result_queue.put(work_result)
    return


def multiprocess_data_gen(num_workers_in, number_of_data_in):
    number_of_works = int(number_of_data_in/num_workers_in)
    pool = Pool(num_workers_in)
    m = Manager()
    result_queue = m.Queue()
    pool.starmap(work, [(number_of_works, result_queue) for _ in range(num_workers_in)])
    pool.close()
    pool.join()
    return result_queue


if __name__ == "__main__":
    num_workers = args.num_workers
    number_of_data = args.num_data
    if os.path.exists('kf_train_noised.csv') is False:
        # train data
        data = multiprocess_data_gen(num_workers, number_of_data)
        data.put("stop")
        res = []
        while True:
            data_piece = data.get()
            if data_piece == "stop":
                break
            else:
                res = res + data_piece
        res = np.array(res)
        df = pd.DataFrame(res[:, :-1])
        df.to_csv('kf_train_noised.csv', header=False, index=False)
        print("Train data generation complete")
    if os.path.exists('kf_val_noised.csv') is False:
        # validation data
        data = multiprocess_data_gen(num_workers, int(number_of_data/10))
        data.put("stop")
        res = []
        while True:
            data_piece = data.get()
            if data_piece == "stop":
                break
            else:
                res = res + data_piece
        res = np.array(res)
        df = pd.DataFrame(res[:, :-1])
        df.to_csv('kf_val_noised.csv', header=False, index=False)
        print("validation data generation complete")
