import numpy as np
import pandas as pd
import os
from data_gen import uni_data_generator
from multiprocessing import Process, Queue


def work(id, number_of_loops, result_queue):
    work_result = uni_data_generator(number_of_loops, True)
    result_queue.put(work_result)
    return


def multiprocess_data_gen(num_workers_in, number_of_data_in):
    workers = []
    number_of_works = int(number_of_data_in/num_workers_in)
    workers_index = np.arange(num_workers_in)
    result = Queue()

    for thread_id in workers_index:
        worker = Process(target=work, args=(thread_id, number_of_works, result))
        workers.append(worker)
        worker.start()

    for worker in workers:
        worker.join()

    return result


if __name__ == "__main__":
    num_workers = 60
    number_of_data = 300000
    if os.path.exists('kf_train.csv') is False:
        # train data
        data = multiprocess_data_gen(num_workers, number_of_data)
        data = np.array(data)
        df = pd.DataFrame(data[:, :-1])
        df.to_csv('kf_train.csv', header=False, index=False)
        print("Train data generation complete")
    if os.path.exists('kf_val.csv') is False:
        # validation data
        data = multiprocess_data_gen(num_workers, number_of_data)
        data = np.array(data)
        df = pd.DataFrame(data[:, :-1])
        df.to_csv('kf_train.csv', header=False, index=False)
        print("Train data generation complete")
