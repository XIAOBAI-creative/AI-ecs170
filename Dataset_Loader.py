'''
Dataset loader for ECS 170 Stage 3 handwritten digit image classification.
It supports common formats:
1. image folders: train/0/*.png, test/0/*.png, or 0/*.png
2. npz files: x_train/y_train/x_test/y_test or X/y
3. csv/txt files: pixel columns followed by label column
'''

from local_code.base_class.dataset import dataset
import os
import csv
import math
import numpy as np

try:
    from PIL import Image
except ImportError:
    Image = None


class Dataset_Loader(dataset):
    data = None
    dataset_source_folder_path = None
    dataset_source_file_name = None
    image_size = (28, 28)
    convert_gray = True
    normalize = True

    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)

    def load(self):
        print('loading stage 3 digit data...')
        path = self._get_source_path()
        print('dataset path:', path)

        if os.path.isdir(path):
            data = self._load_from_folder(path)
        elif path.endswith('.npz'):
            data = self._load_from_npz(path)
        elif path.endswith('.npy'):
            raise ValueError('Single .npy file is ambiguous. Use .npz with X/y or train/test arrays.')
        elif path.endswith('.csv') or path.endswith('.txt'):
            data = self._load_from_table(path)
        else:
            raise ValueError('Unsupported dataset format: ' + path)

        data = self._normalize_output(data)
        self._print_summary(data)
        return data

    def _get_source_path(self):
        folder = self.dataset_source_folder_path or ''
        name = self.dataset_source_file_name or ''
        path = os.path.join(folder, name)
        path = os.path.normpath(path)
        return path

    def _load_from_folder(self, root):
        train_dir = self._find_existing_dir(root, ['train', 'training', 'Train', 'Training'])
        test_dir = self._find_existing_dir(root, ['test', 'testing', 'Test', 'Testing'])

        if train_dir is not None and test_dir is not None:
            X_train, y_train = self._read_image_class_folder(train_dir)
            X_test, y_test = self._read_image_class_folder(test_dir)
            return {'train': {'X': X_train, 'y': y_train}, 'test': {'X': X_test, 'y': y_test}}

        X, y = self._read_image_class_folder(root)
        return {'X': X, 'y': y}

    def _find_existing_dir(self, root, names):
        for name in names:
            path = os.path.join(root, name)
            if os.path.isdir(path):
                return path
        return None

    def _read_image_class_folder(self, root):
        if Image is None:
            raise ImportError('Pillow is required for image folders. Run: pip install pillow')

        image_ext = {'.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tif', '.tiff'}
        X = []
        y = []

        class_names = [d for d in sorted(os.listdir(root)) if os.path.isdir(os.path.join(root, d))]
        if len(class_names) == 0:
            image_files = []
            for dirpath, _, filenames in os.walk(root):
                for filename in filenames:
                    ext = os.path.splitext(filename)[1].lower()
                    if ext in image_ext:
                        image_files.append(os.path.join(dirpath, filename))
            if len(image_files) == 0:
                raise ValueError('No image files found under ' + root)
            raise ValueError('Images were found, but no class folders were found. Expected folders like 0/, 1/, 2/, ...')

        label_map = {name: i for i, name in enumerate(class_names)}
        numeric_labels = all(name.isdigit() for name in class_names)

        for class_name in class_names:
            class_path = os.path.join(root, class_name)
            label = int(class_name) if numeric_labels else label_map[class_name]
            for dirpath, _, filenames in os.walk(class_path):
                for filename in sorted(filenames):
                    ext = os.path.splitext(filename)[1].lower()
                    if ext not in image_ext:
                        continue
                    img_path = os.path.join(dirpath, filename)
                    X.append(self._read_one_image(img_path))
                    y.append(label)

        if len(X) == 0:
            raise ValueError('Class folders exist, but no image files were loaded from ' + root)
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)

    def _read_one_image(self, img_path):
        img = Image.open(img_path)
        if self.convert_gray:
            img = img.convert('L')
        else:
            img = img.convert('RGB')
        if self.image_size is not None:
            img = img.resize(self.image_size)
        arr = np.array(img, dtype=np.float32)
        if self.normalize:
            arr = arr / 255.0
        if self.convert_gray:
            arr = arr.reshape(1, arr.shape[0], arr.shape[1])
        else:
            arr = arr.transpose(2, 0, 1)
        return arr

    def _load_from_npz(self, path):
        obj = np.load(path, allow_pickle=True)
        keys = set(obj.files)

        train_key_sets = [
            ('x_train', 'y_train', 'x_test', 'y_test'),
            ('X_train', 'y_train', 'X_test', 'y_test'),
            ('train_X', 'train_y', 'test_X', 'test_y'),
        ]
        for xtr, ytr, xte, yte in train_key_sets:
            if {xtr, ytr, xte, yte}.issubset(keys):
                return {'train': {'X': obj[xtr], 'y': obj[ytr]}, 'test': {'X': obj[xte], 'y': obj[yte]}}

        xy_key_sets = [('X', 'y'), ('x', 'y'), ('data', 'label'), ('data', 'labels')]
        for xkey, ykey in xy_key_sets:
            if {xkey, ykey}.issubset(keys):
                return {'X': obj[xkey], 'y': obj[ykey]}

        raise ValueError('Cannot recognize npz keys: ' + str(obj.files))

    def _load_from_table(self, path):
        rows = []
        delimiter = ',' if path.endswith('.csv') else None
        with open(path, 'r') as f:
            reader = csv.reader(f, delimiter=delimiter) if delimiter else None
            if reader is not None:
                for row in reader:
                    if len(row) > 0:
                        rows.append(row)
            else:
                for line in f:
                    line = line.strip()
                    if line:
                        rows.append(line.replace(',', ' ').split())

        clean_rows = []
        for row in rows:
            try:
                clean_rows.append([float(v) for v in row])
            except ValueError:
                continue

        arr = np.array(clean_rows, dtype=np.float32)
        if arr.shape[1] < 2:
            raise ValueError('Table file needs pixel columns plus one label column.')
        X = arr[:, :-1]
        y = arr[:, -1].astype(np.int64)
        X = self._reshape_flat_images(X)
        return {'X': X, 'y': y}

    def _reshape_flat_images(self, X):
        side = int(math.sqrt(X.shape[1]))
        if side * side != X.shape[1]:
            raise ValueError('Flat pixel count is not a square number: ' + str(X.shape[1]))
        X = X.reshape(X.shape[0], 1, side, side)
        if self.normalize and X.max() > 1.0:
            X = X / 255.0
        return X.astype(np.float32)

    def _normalize_output(self, data):
        if 'train' in data and 'test' in data:
            data['train']['X'] = self._fix_X_shape(data['train']['X'])
            data['test']['X'] = self._fix_X_shape(data['test']['X'])
            data['train']['y'] = np.array(data['train']['y'], dtype=np.int64).reshape(-1)
            data['test']['y'] = np.array(data['test']['y'], dtype=np.int64).reshape(-1)
            return data
        data['X'] = self._fix_X_shape(data['X'])
        data['y'] = np.array(data['y'], dtype=np.int64).reshape(-1)
        return data

    def _fix_X_shape(self, X):
        X = np.array(X, dtype=np.float32)
        if X.max() > 1.0 and self.normalize:
            X = X / 255.0
        if len(X.shape) == 2:
            X = self._reshape_flat_images(X)
        elif len(X.shape) == 3:
            X = X.reshape(X.shape[0], 1, X.shape[1], X.shape[2])
        elif len(X.shape) == 4:
            if X.shape[-1] in [1, 3] and X.shape[1] not in [1, 3]:
                X = X.transpose(0, 3, 1, 2)
            if self.convert_gray and X.shape[1] == 3:
                X = X.mean(axis=1, keepdims=True)
        else:
            raise ValueError('Unsupported X shape: ' + str(X.shape))
        return X.astype(np.float32)

    def _print_summary(self, data):
        if 'train' in data and 'test' in data:
            print('train X shape:', data['train']['X'].shape, 'train y shape:', data['train']['y'].shape)
            print('test X shape:', data['test']['X'].shape, 'test y shape:', data['test']['y'].shape)
            print('labels:', sorted(set(data['train']['y'].tolist() + data['test']['y'].tolist())))
        else:
            print('X shape:', data['X'].shape, 'y shape:', data['y'].shape)
            print('labels:', sorted(set(data['y'].tolist())))
