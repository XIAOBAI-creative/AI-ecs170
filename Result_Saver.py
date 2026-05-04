'''
Result saver for Stage 3.
'''

from local_code.base_class.result import result
import pickle
import os


class Result_Saver(result):
    data = None
    fold_count = None
    result_destination_folder_path = None
    result_destination_file_name = None

    def save(self):
        print('saving results...')
        os.makedirs(self.result_destination_folder_path, exist_ok=True)
        suffix = '' if self.fold_count is None else '_' + str(self.fold_count)
        path = self.result_destination_folder_path + self.result_destination_file_name + suffix
        with open(path, 'wb') as f:
            pickle.dump(self.data, f)
        print('saved result to:', path)
