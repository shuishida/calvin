import json
from collections import defaultdict, OrderedDict
import os
import torch


class JSONIO(dict):
    def __init__(self, path=None):
        self.filepath = path
        if path and os.path.isfile(path):
            with open(path, "r") as f:
                super(JSONIO, self).__init__(json.load(f))
        else:
            super(JSONIO, self).__init__()

    def __setitem__(self, key, value):
        super(JSONIO, self).__setitem__(str(key), value)

    def __getitem__(self, index):
        return super(JSONIO, self).__getitem__(str(index))

    def save(self):
        if self.filepath:
            with open(self.filepath, "w") as f:
                json.dump(self, f)


class DictIO(dict):
    def __init__(self, data=None, path=None):
        self.path = path
        if path:
            dirpath = os.path.dirname(path)
            if not os.path.exists(dirpath):
                os.makedirs(dirpath)
            if data is None and os.path.exists(path):
                data = torch.load(path)
        if data is None: data = {}
        super(DictIO, self).__init__(data)

    def save(self):
        if self.path:
            torch.save(dict(self), self.path)

    def delete(self, key):
        del self[key]


class MultiFileIO:
    def __init__(self, dirpath, cash_size: int = None):
        self.dir = dirpath
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)
        self._data = self.files()
        self._len = len(self._data.keys())
        assert cash_size is None or cash_size > 0
        self.cash_size = cash_size
        self.cash_order = []

    def files(self):
        return {d.replace(".pt", ""): None for d in os.listdir(self.dir)
                if os.path.isfile(os.path.join(self.dir, d)) and d.endswith(".pt")}

    def file(self, key):
        return os.path.join(self.dir, f"{key}.pt")

    def __len__(self):
        return self._len

    def __setitem__(self, key, value):
        if key not in self._data:
            self._len += 1
            self._data[key] = None
        if self.cash_size:
            self._update_cash(key, value)
        else:
            torch.save(value, self.file(key))

    def __getitem__(self, item):
        val = self._data[item]
        if val is None:
            try:
                val = torch.load(self.file(item))
            except Exception as e:
                print("Failed to load ", self.file(item))
                print(e)
            if self.cash_size:
                self._update_cash(item, val)
        return val

    def delete(self, key):
        del self._data[key]
        if key in self.cash_order: self.cash_order.remove(key)
        self._len -= 1
        path = self.file(key)
        if os.path.exists(path): os.remove(path)

    def _update_cash(self, key, value):
        if key not in self.cash_order:
            remove_keys, keep_keys = self.cash_order[:-self.cash_size], self.cash_order[-self.cash_size:]
            for k in remove_keys:
                self._save(k)
                self._data[k] = None
            self.cash_order = keep_keys + [key]
        self._data[key] = value

    def _save(self, key):
        val = self._data.get(key)
        if val is not None:
            torch.save(self._data[key], self.file(key))

    def save(self):
        for key, val in self._data.items():
            self._save(key)
