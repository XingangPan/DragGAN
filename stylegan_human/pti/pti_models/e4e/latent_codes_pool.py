import random
import torch


class LatentCodesPool:
    """This class implements latent codes buffer that stores previously generated w latent codes.
    This buffer enables us to update discriminators using a history of generated w's
    rather than the ones produced by the latest encoder.
    """

    def __init__(self, pool_size):
        """Initialize the ImagePool class
        Parameters:
            pool_size (int) -- the size of image buffer, if pool_size=0, no buffer will be created
        """
        self.pool_size = pool_size
        if self.pool_size > 0:  # create an empty pool
            self.num_ws = 0
            self.ws = []

    def query(self, ws):
        """Return w's from the pool.
        Parameters:
            ws: the latest generated w's from the generator
        Returns w's from the buffer.
        By 50/100, the buffer will return input w's.
        By 50/100, the buffer will return w's previously stored in the buffer,
        and insert the current w's to the buffer.
        """
        if self.pool_size == 0:  # if the buffer size is 0, do nothing
            return ws
        return_ws = []
        for w in ws:  # ws.shape: (batch, 512) or (batch, n_latent, 512)
            # w = torch.unsqueeze(image.data, 0)
            if w.ndim == 2:
                i = random.randint(0, len(w) - 1)  # apply a random latent index as a candidate
                w = w[i]
            self.handle_w(w, return_ws)
        return_ws = torch.stack(return_ws, 0)   # collect all the images and return
        return return_ws

    def handle_w(self, w, return_ws):
        if self.num_ws < self.pool_size:  # if the buffer is not full; keep inserting current codes to the buffer
            self.num_ws = self.num_ws + 1
            self.ws.append(w)
            return_ws.append(w)
        else:
            p = random.uniform(0, 1)
            if p > 0.5:  # by 50% chance, the buffer will return a previously stored latent code, and insert the current code into the buffer
                random_id = random.randint(0, self.pool_size - 1)  # randint is inclusive
                tmp = self.ws[random_id].clone()
                self.ws[random_id] = w
                return_ws.append(tmp)
            else:  # by another 50% chance, the buffer will return the current image
                return_ws.append(w)
