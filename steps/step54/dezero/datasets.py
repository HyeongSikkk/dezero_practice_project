import numpy as np
from dezero.utils import get_file
import gzip

class Dataset :
    def __init__(self, train = True, transform = None, target_transform = None) :
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        if self.transform is None :
            self.transform = lambda x : x
        if self.target_transform is None :
            self.target_transform = lambda x : x
                        
        self.data = None
        self.label = None
        self.prepare()
        
    def __getitem__(self, index) :
        assert np.isscalar(index) # index는 정수(스칼라)만 지원
        if self.label is None :
            return self.transform(self.data[index]), None
        else :
            return self.transform(self.data[index]),\
                self.target_transform(self.label[index])
        
    def __len__(self) :
        return len(self.data)
    
    def prepare(self) :
        pass
    

# class Spiral(Dataset) :
#     def prepare(self) :
#         self.data, self.label = get_spiral(self.train)


class BigData(Dataset) :
    def __getitem__(index) :
        x = np.load(f"data/{index}.npy")
        t = np.load(f"label/{index}.npy")
        return x, t
    
    def __len__() :
        return 1000000
    

class Normalize:
    """Normalize a NumPy array with mean and standard deviation.

    Args:
        mean (float or sequence): mean for all values or sequence of means for
         each channel.
        std (float or sequence):
    """
    def __init__(self, mean=0, std=1):
        self.mean = mean
        self.std = std

    def __call__(self, array):
        mean, std = self.mean, self.std

        if not np.isscalar(mean):
            mshape = [1] * array.ndim
            mshape[0] = len(array) if len(self.mean) == 1 else len(self.mean)
            mean = np.array(self.mean, dtype=array.dtype).reshape(*mshape)
        if not np.isscalar(std):
            rshape = [1] * array.ndim
            rshape[0] = len(array) if len(self.std) == 1 else len(self.std)
            std = np.array(self.std, dtype=array.dtype).reshape(*rshape)
        return (array - mean) / std

class Flatten:
    """Flatten a NumPy array.
    """
    def __call__(self, array):
        return array.flatten()

class AsType:
    def __init__(self, dtype=np.float32):
        self.dtype = dtype

    def __call__(self, array):
        return array.astype(self.dtype)


ToFloat = AsType

class Compose:
    """Compose several transforms.

    Args:
        transforms (list): list of transforms
    """
    def __init__(self, transforms=[]):
        self.transforms = transforms

    def __call__(self, img):
        if not self.transforms:
            return img
        for t in self.transforms:
            img = t(img)
        return img

class MNIST(Dataset):

    def __init__(self, train=True,
                 transform=Compose([Flatten(), ToFloat(),
                                    Normalize(0., 255.)]),
                 target_transform=None):
        super().__init__(train, transform, target_transform)

    def prepare(self):
        url = 'http://yann.lecun.com/exdb/mnist/'
        train_files = {'target': 'train-images-idx3-ubyte.gz',
                       'label': 'train-labels-idx1-ubyte.gz'}
        test_files = {'target': 't10k-images-idx3-ubyte.gz',
                      'label': 't10k-labels-idx1-ubyte.gz'}

        files = train_files if self.train else test_files
        data_path = get_file(url + files['target'])
        label_path = get_file(url + files['label'])

        self.data = self._load_data(data_path)
        self.label = self._load_label(label_path)

    def _load_label(self, filepath):
        with gzip.open(filepath, 'rb') as f:
            labels = np.frombuffer(f.read(), np.uint8, offset=8)
        return labels

    def _load_data(self, filepath):
        with gzip.open(filepath, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        data = data.reshape(-1, 1, 28, 28)
        return data

    # def show(self, row=10, col=10):
        # H, W = 28, 28
        # img = np.zeros((H * row, W * col))
        # for r in range(row):
            # for c in range(col):
                # img[r * H:(r + 1) * H, c * W:(c + 1) * W] = self.data[
                    # np.random.randint(0, len(self.data) - 1)].reshape(H, W)
        # plt.imshow(img, cmap='gray', interpolation='nearest')
        # plt.axis('off')
        # plt.show()

    @staticmethod
    def labels():
        return {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9'}