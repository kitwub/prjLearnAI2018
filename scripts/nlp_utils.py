import numpy as np

import chainer
from chainer import cuda


def convert_seq(batch, device=None, with_label=True):
    def to_device_batch(batch, label=False):
        if device is None:
            return batch
        elif device < 0:
            if label:
                return chainer.dataset.to_device(device, np.array(batch, dtype=np.int32))
            else:
                return [[chainer.dataset.to_device(device, y) for y in x] for x in batch]
        else:
            if label:
                return chainer.dataset.to_device(device, np.array(batch, dtype=np.int32))
            else:
                xp = cuda.cupy.get_array_module(*batch)
                doc_len_list = []
                concat_doc = []
                for doc in batch:
                    sent_len_list = [len(sent) for sent in doc]
                    doc_len_list.append(sent_len_list)
                    concat_doc.append(xp.concatenate(doc, axis=0))
                concat = xp.concatenate(concat_doc, axis=0)
                concat_dev = chainer.dataset.to_device(device, concat)
                doc_dev = xp.split(concat_dev, np.cumsum([np.sum(x) for x in doc_len_list[:-1]], dtype=np.int32))
                batch_dev = []
                for i in range(len(doc_dev)):
                    batch_dev.append(xp.split(doc_dev[i], np.cumsum(doc_len_list[i][:-1], dtype=np.int32)))
                return batch_dev

    if with_label:
        return to_device_batch([x for x, _ in batch], label=False), to_device_batch([y for _, y in batch], label=True)
    else:
        return to_device_batch([x for x in batch])
