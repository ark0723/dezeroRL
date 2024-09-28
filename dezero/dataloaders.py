import math
import numpy as np
from dezero import cuda


class DataLoader:
    def __init__(self, dataset, batch_size, shuffle=True, gpu=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.data_size = len(dataset)
        self.max_iter = math.ceil(self.data_size / batch_size)
        self.gpu = gpu
        self.reset()

    def reset(self):
        self.iteration = 0  # 반복횟수 초기화
        if self.shuffle:
            self.index = np.random.permutation(len(self.dataset))
        else:
            self.index = np.arange(len(self.dataset))

    def __iter__(self):
        return self

    def __next__(self):
        if self.iteration >= self.max_iter:
            self.reset()
            raise StopIteration

        i, batch_size = self.iteration, self.batch_size
        batch_idx = self.index[i * batch_size : (i + 1) * batch_size]
        batch = [self.dataset[i] for i in batch_idx]

        xp = cuda.cupy if self.gpu else np
        batch_x = xp.array([example[0] for example in batch])
        batch_t = xp.array([example[1] for example in batch])

        self.iteration += 1
        return batch_x, batch_t

    def to_cpu(self):
        self.gpu = False

    def to_gpu(self):
        self.gpu = True


class SeqDataLoader(DataLoader):
    def __init__(self, dataset, batch_size, gpu=False):
        super().__init__(dataset=dataset, batch_size=batch_size, shuffle=False, gpu=gpu)

    def __next__(self):
        if self.iteration >= self.max_iter:
            self.reset()
            raise StopIteration

        jump = self.data_size // self.batch_size
        batch_idx = [
            (i * jump + self.iteration) % self.data_size for i in range(self.batch_size)
        ]
        batch = [self.dataset[i] for i in batch_idx]

        xp = cuda.cupy if self.gpu else np
        x = xp.array([data[0] for data in batch])
        t = xp.array([data[1] for data in batch])

        self.iteration += 1
        return x, t
