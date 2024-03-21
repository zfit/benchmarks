from memory_profiler import profile
import numpy as np
import tensorflow as tf

# @tf.function(autograph=False)
# @profile
def load_data():
    # data_np = np.random.normal(size=100000000)
    data_np = tf.random.normal(shape=(80, 1)) + 1
    dataset = tf.data.Dataset.from_tensor_slices(data_np)
    # dataset = tf.data.Dataset.range(100)
    dataset2 = dataset.batch(3)
    for data in dataset2:
        print(data)
        sqrt = tf.square(data)
    for data in dataset2:
        print(data)
        sqrt = tf.square(data)
        # print(sqrt)



    # dataset = tf.convert_to_tensor(data_np)


if __name__ == '__main__':
    load_data()