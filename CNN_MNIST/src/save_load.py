# save_load

import tensorflow as tf

def save(s, path):
    Save=tf.train.Saver()
    Save.save(s, path)
    pass
