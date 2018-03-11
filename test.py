# -*- coding: utf-8 -*-
# /usr/bin/python2
'''
By kyubyong park. kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/sudoku
'''
from __future__ import print_function
import tensorflow as tf
import numpy as np
from train import Graph
from data_load import load_data
from hyperparams import Hyperparams as hp
import os
import sys


def print_result(x, y, preds):
    total_hits, total_blanks = 0, 0
    for xx, yy, pp in zip(x.reshape(-1, 9 * 9), y.reshape(-1, 9 * 9), preds.reshape(-1, 9 * 9)):  # sample-wise

        expected = yy[xx == 0]
        got = pp[xx == 0]

        num_hits = np.equal(expected, got).sum()
        num_blanks = len(expected)

        total_hits += num_hits
        total_blanks += num_blanks

    print("Total accuracy = %d/%d = %.2f" % (total_hits, total_blanks, float(total_hits) / total_blanks))


def write_to_file(x, y, preds, fout):
    '''Writes to file.
    Args:
      x: A 3d array with shape of [N, 9, 9]. Quizzes where blanks are represented as 0's.
      y: A 3d array with shape of [N, 9, 9]. Solutions.
      preds: A 3d array with shape of [N, 9, 9]. Predictions.
      fout: A string. File path of the output file where the results will be written.
    '''
    with open(fout, 'w') as fout:
        total_hits, total_blanks = 0, 0
        for xx, yy, pp in zip(x.reshape(-1, 9 * 9), y.reshape(-1, 9 * 9), preds.reshape(-1, 9 * 9)):  # sample-wise
            fout.write("qz: {}\n".format("".join(str(num) if num != 0 else "_" for num in xx)))
            fout.write("sn: {}\n".format("".join(str(num) for num in yy)))
            fout.write("pd: {}\n".format("".join(str(num) for num in pp)))

            expected = yy[xx == 0]
            got = pp[xx == 0]

            num_hits = np.equal(expected, got).sum()
            num_blanks = len(expected)

            fout.write("accuracy = %d/%d = %.2f\n\n" % (num_hits, num_blanks, float(num_hits) / num_blanks))

            total_hits += num_hits
            total_blanks += num_blanks
        fout.write("Total accuracy = %d/%d = %.2f\n\n" % (total_hits, total_blanks, float(total_hits) / total_blanks))


def test(test_input_path, num_threads=4):
    x, y = load_data(fpath=test_input_path)

    g = Graph(is_training=False)
    with g.graph.as_default():
        sv = tf.train.Supervisor()
        with sv.managed_session(config=tf.ConfigProto(allow_soft_placement=True,
                                                      intra_op_parallelism_threads=num_threads)) as sess:
            # Restore parameters
            sv.saver.restore(sess, tf.train.latest_checkpoint(hp.logdir))

            # Get model name
            mname = open(hp.logdir + '/checkpoint', 'r').read().split('"')[1]  # model name

            import copy
            _preds = copy.copy(x)
            while 1:
                istarget, probs, preds = sess.run([g.istarget, g.probs, g.preds], {g.x: _preds, g.y: y})
                probs = probs.astype(np.float32)
                preds = preds.astype(np.float32)

                probs *= istarget  # (N, 9, 9)
                preds *= istarget  # (N, 9, 9)

                probs = np.reshape(probs, (-1, 9 * 9))  # (N, 9*9)
                preds = np.reshape(preds, (-1, 9 * 9))  # (N, 9*9)

                _preds = np.reshape(_preds, (-1, 9 * 9))
                maxprob_ids = np.argmax(probs, axis=1)  # (N, ) <- blanks of the most probable prediction
                maxprobs = np.max(probs, axis=1, keepdims=False)
                for j, (maxprob_id, maxprob) in enumerate(zip(maxprob_ids, maxprobs)):
                    if maxprob != 0:
                        _preds[j, maxprob_id] = preds[j, maxprob_id]
                _preds = np.reshape(_preds, (-1, 9, 9))
                _preds = np.where(x == 0, _preds, y)  # # Fill in the non-blanks with correct numbers

                if np.count_nonzero(_preds) == _preds.size: break

            print_result(x.astype(np.int32), y, _preds.astype(np.int32))


if __name__ == '__main__':
    test_file = sys.argv[1]
    num_thread = int(sys.argv[2])
    tf.logging.set_verbosity(tf.logging.ERROR)
    test(test_file, num_thread)
