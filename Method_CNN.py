'''
CNN model for ECS 170 Stage 3 handwritten digit classification.
This model expects grayscale input with shape N x 1 x H x W and predicts 10 classes.
'''

from local_code.base_class.method import method
from local_code.stage_3_code.Evaluate_Accuracy import Evaluate_Accuracy
import os
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader


class Method_CNN(method, nn.Module):
    data = None
    max_epoch = 20
    learning_rate = 0.001
    batch_size = 64
    num_classes = 10
    history_destination_folder_path = '../../result/stage_3_result/'

    def __init__(self, mName, mDescription):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)

        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc1 = nn.Linear(64 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, self.num_classes)
        self.history = []

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = self.adaptive_pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

    def _prepare_X(self, X):
        X = torch.FloatTensor(np.array(X, dtype=np.float32))
        if len(X.shape) == 3:
            X = X.unsqueeze(1)
        if len(X.shape) == 4 and X.shape[-1] in [1, 3] and X.shape[1] not in [1, 3]:
            X = X.permute(0, 3, 1, 2)
        if len(X.shape) == 4 and X.shape[1] == 3:
            X = X.mean(dim=1, keepdim=True)
        return X

    def _prepare_y(self, y):
        return torch.LongTensor(np.array(y, dtype=np.int64).reshape(-1))

    def _accuracy(self, logits, y):
        pred = logits.max(1)[1]
        return (pred == y).float().mean().item()

    def train_model(self, X_train, y_train, X_test, y_test):
        X_train = self._prepare_X(X_train)
        y_train = self._prepare_y(y_train)
        X_test = self._prepare_X(X_test)
        y_test = self._prepare_y(y_test)

        found_classes = int(torch.max(torch.cat([y_train, y_test])).item()) + 1
        if found_classes != self.num_classes:
            self.num_classes = found_classes
            self.fc2 = nn.Linear(128, self.num_classes)

        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        loss_function = nn.CrossEntropyLoss()
        train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=self.batch_size, shuffle=True)

        self.history = []
        for epoch in range(1, self.max_epoch + 1):
            self.train()
            total_loss = 0.0
            total_count = 0

            for batch_X, batch_y in train_loader:
                y_pred = self.forward(batch_X)
                train_loss = loss_function(y_pred, batch_y)

                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()

                total_loss += train_loss.item() * batch_X.shape[0]
                total_count += batch_X.shape[0]

            self.eval()
            with torch.no_grad():
                train_logits = self.forward(X_train)
                test_logits = self.forward(X_test)
                train_acc = self._accuracy(train_logits, y_train)
                test_acc = self._accuracy(test_logits, y_test)
                avg_loss = total_loss / total_count

            row = {'epoch': epoch, 'train_loss': avg_loss, 'train_accuracy': train_acc, 'test_accuracy': test_acc}
            self.history.append(row)
            print('Epoch:', epoch, 'Loss:', round(avg_loss, 4), 'Train Acc:', round(train_acc, 4), 'Test Acc:', round(test_acc, 4))

        self._save_history()

    def test(self, X):
        X = self._prepare_X(X)
        self.eval()
        with torch.no_grad():
            y_pred = self.forward(X)
        return y_pred.max(1)[1].numpy()

    def run(self):
        print('method running...')
        print('--start training...')
        self.train_model(
            self.data['train']['X'],
            self.data['train']['y'],
            self.data['test']['X'],
            self.data['test']['y']
        )
        print('--start testing...')
        pred_y = self.test(self.data['test']['X'])
        true_y = np.array(self.data['test']['y'], dtype=np.int64).reshape(-1)
        return {'pred_y': pred_y, 'true_y': true_y, 'history': self.history}

    def _save_history(self):
        os.makedirs(self.history_destination_folder_path, exist_ok=True)
        csv_path = os.path.join(self.history_destination_folder_path, 'digit_cnn_learning_curve.csv')
        with open(csv_path, 'w') as f:
            f.write('epoch,train_loss,train_accuracy,test_accuracy\n')
            for row in self.history:
                f.write(str(row['epoch']) + ',' + str(row['train_loss']) + ',' + str(row['train_accuracy']) + ',' + str(row['test_accuracy']) + '\n')
        print('saved learning curve data to:', csv_path)

        try:
            import matplotlib.pyplot as plt
            epochs = [row['epoch'] for row in self.history]
            losses = [row['train_loss'] for row in self.history]
            plt.figure()
            plt.plot(epochs, losses)
            plt.xlabel('Epoch')
            plt.ylabel('Training Loss')
            plt.title('Digit CNN Training Loss')
            plt.savefig(os.path.join(self.history_destination_folder_path, 'digit_cnn_training_loss.png'), bbox_inches='tight')
            plt.close()

            train_acc = [row['train_accuracy'] for row in self.history]
            test_acc = [row['test_accuracy'] for row in self.history]
            plt.figure()
            plt.plot(epochs, train_acc, label='Train Accuracy')
            plt.plot(epochs, test_acc, label='Test Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.title('Digit CNN Accuracy')
            plt.legend()
            plt.savefig(os.path.join(self.history_destination_folder_path, 'digit_cnn_accuracy.png'), bbox_inches='tight')
            plt.close()
            print('saved learning curve plots to:', self.history_destination_folder_path)
        except Exception as e:
            print('plot saving skipped:', e)
