import os
import glob
import subprocess
import numpy as np
import tensorflow as tf
from time import time, ctime


class Statistic(object):
    """
    Class for keeping track of statistics, saving model, etc.
    """

    def __init__(self, sess, t_test, t_save, t_learn_start, run_dir, save_dir, variables, load, chtc, window_length, termination_p_hat, max_to_keep=2):
        self.sess = sess
        self.t_test = t_test
        self.t_save = t_save
        self.t_start = 0
        self.t_learn_start = t_learn_start

        self.has_updated = False

        with tf.variable_scope('t'):
            self.t_op = tf.Variable(0, trainable=False, name='t')
            self.t_add_op = self.t_op.assign_add(1)

        self.run_dir = run_dir
        self.save_dir = save_dir
        self.model_dir = save_dir + "data/checkpoints/"
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
            
        self.log_dir = save_dir + "data/logs/"
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        self.log_filename = self.log_dir + 'log.csv'
        self.load = load
        self.chtc = chtc
        self.window_length = window_length
        self.termination_p_hat = termination_p_hat
        
        # Saver saves periodically.
        self.saver = tf.train.Saver(list(variables) + [self.t_op], max_to_keep=max_to_keep)
        # Saver saves when training is halted.
        self.latest_saver = tf.train.Saver(list(variables) + [self.t_op], max_to_keep=1)

        self.rewards = np.array([])
        self.game_length = np.array([])

    def on_step(self, t, is_update):
        """
        After a step, see if model should be saved and increment iteration count.
        """        
        if t >= self.t_learn_start + self.t_start:
            if is_update:
                self.has_updated = True
                
            if t % self.t_save == 0 and self.has_updated:
                # Save model.
                self.save_model(t, self.saver)

        # Increment iteration count.
        self.t_add_op.eval(session=self.sess)

    def get_t(self):
        """
        Get t.
        """
        return self.t_op.eval(session=self.sess)

    def save_model(self, t, saver):
        """
        Save model.
        """
        saver.save(self.sess, self.model_dir, global_step=t)

    def load_model(self):
        """
        Load model.
        """
        # Get checkpoint name.
        try:
            ckpt = tf.train.get_checkpoint_state(self.model_dir)
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            fname = os.path.join(self.model_dir, ckpt_name)

            # Restore model.
            self.saver.restore(self.sess, fname)

            print(" [*] Load SUCCESS: %s" % fname)
            return True
        except:
            print(" [!] Load FAILED: %s" % self.model_dir)
            print(os.listdir(self.model_dir))

            # In CHTC mode, it is okay to not successfully load.
            if not self.chtc:
                exit(1)

            return False

    def write_log(self, log_filename, data):
        """
        Writes a string to log.
        """
        open(log_filename, 'a').write(str(data))

    def clear_log(self, log_filename):
        """
        Clears log.
        """
        open(log_filename, 'w').close()

    def upload_log(self, log_filename):
        call_subprocess(('gdrive','update','1APM_EBWdkKncwanhq0m4odg4lu1GmCa6', log_filename))

    def zip_data(self, clean):
        """
        Zips data directory.
        """
        if clean:
            # Remove checkpoints that were saved when the program received an interrupt signal.
            for file in glob.glob(self.save_dir + 'data/checkpoints/*'):
                try:
                    ckpt = file.replace(self.save_dir + 'data/checkpoints/','').replace('-','')
                    ckpt_number = ckpt[:ckpt.find('.')]
                    i = int(ckpt_number)
                    if i % self.t_save != 0:
                        os.remove(file)
                except:
                    pass
    
        call_subprocess(('tar','czf', self.save_dir + 'output_data.tar.gz.tmp', 'data', '-C', self.save_dir))
        call_subprocess(('mv', self.save_dir + 'output_data.tar.gz.tmp', self.save_dir + 'output_data.tar.gz'))

    def evaluate_termination_criteria(self):
        with open(self.log_filename,'r') as f:
            data = [line.strip() for line in f.readlines()]
        
        p = [float(line.split(',')[5].strip()) for line in data]
        if len(p) >= self.window_length:
            # Mean over sliding window of past average scores.
            window_mean = np.mean(p[-self.window_length:])
            if window_mean >= self.termination_p_hat:
                raise SystemExit("Termination criteria achieved. window p_hat: {}".format(window_mean))

def call_subprocess(call):
    """
    Wrapper for calling a subprocess.
    """
    subprocess.call(call, stdout=open(os.devnull, 'wb'), stderr=open(os.devnull, 'wb'))
