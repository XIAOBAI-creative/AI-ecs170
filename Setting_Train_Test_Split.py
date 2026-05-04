'''
Train/test setting for Stage 3. If the dataset already has train/test splits,
this file uses them directly. Otherwise it makes a random split.
'''

from local_code.base_class.setting import setting
from sklearn.model_selection import train_test_split


class Setting_Train_Test_Split(setting):
    test_size = 0.2
    random_state = 2

    def load_run_save_evaluate(self):
        loaded_data = self.dataset.load()

        if 'train' in loaded_data and 'test' in loaded_data:
            X_train = loaded_data['train']['X']
            y_train = loaded_data['train']['y']
            X_test = loaded_data['test']['X']
            y_test = loaded_data['test']['y']
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                loaded_data['X'],
                loaded_data['y'],
                test_size=self.test_size,
                random_state=self.random_state,
                stratify=loaded_data['y']
            )

        self.method.data = {'train': {'X': X_train, 'y': y_train}, 'test': {'X': X_test, 'y': y_test}}
        learned_result = self.method.run()

        self.result.data = learned_result
        self.result.save()

        self.evaluate.data = learned_result
        return self.evaluate.evaluate(), None
