""" Based on "From Image Hashing to Video Hashing" / Li Weng and Bart Preneel
    + some earlier paper by Weng

    Cumulant-calculation was taken from "http://web.mit.edu/jhawk/mnt/spo/python-lib/src/SciPy_complete-0.3.2/Lib/stats/morestats.py"
        which was written by Travis Oliphant, 2002
"""

import skimage
from skimage.morphology import disk
from skimage.transform import resize
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from bitstring import BitArray, BitStream, pack
from deltasigma import ds_quantize


class FrameHash(object):
    def __init__(self, img):    # image = numpy array + RGB!!!
        self.img = img
        self.preprocess_resize_constants = (512, 384)
        # self.preprocess_gaussian_filter_sigma = 1.0 # ORIG
        # self.preprocess_median_filter_neighborhood = disk(1) # ORIG
        self.preprocess_gaussian_filter_sigma = 2.0
        self.preprocess_median_filter_neighborhood = disk(2)
        self.block_shape = (256, 192)

    def calculate_hash(self):
        self.preprocess()
        self.divide_into_overlapping_blocks()
        self.scan_blocks()
        self.calc_std()
        self.calc_third_order_cumulant()
        self.calc_fourth_order_cumulant()
        self.calc_mean_of_2d_stats()
        self.vector_quantization()

    def preprocess(self):
        self.resize()
        self.convert_to_grey()
        self.gaussian_filter()
        np.clip(self.img, 0, 1, out=self.img)  # due to problems in skimage -> see issue: https://github.com/scikit-image/scikit-image/issues/1529
        self.median_filter()
        self.hist_equalization()

    def divide_into_overlapping_blocks(self):
        i, j = 0, 0
        self.blocks = []
        for i in range(self.block_shape[0], self.preprocess_resize_constants[0]+1, self.block_shape[0]//2):
            for j in range(self.block_shape[1], self.preprocess_resize_constants[1]+1, self.block_shape[1]//2):
                self.blocks.append( self.img[ (i-self.block_shape[0]):i, (j-self.block_shape[1]):j ] )

    def scan_blocks(self):
        self.block_vectors_orig = []
        self.block_vectors_trans = []
        for b in self.blocks:
            self.block_vectors_orig.append( b.flatten() - b.mean())             # normalized!
            self.block_vectors_trans.append( np.transpose(b).flatten() - b.mean() ) # normalized!

    def calc_std(self):
        self.block_orig_std_values = []
        self.block_trans_std_values = []
        for v in self.block_vectors_orig:
            val = np.std(v)
            self.block_orig_std_values.append(val)
        for v in self.block_vectors_trans:
            val = np.std(v)
            self.block_trans_std_values.append(val)

    def calc_third_order_cumulant(self):
        self.block_orig_3rd_order_cumulant_values = []
        self.block_trans_3rd_order_cumulant_values = []
        for v in self.block_vectors_orig:
            val = self.calc_cum(v, 3)
            self.block_orig_3rd_order_cumulant_values.append(val)
        for v in self.block_vectors_trans:
            val = self.calc_cum(v, 3)
            self.block_trans_3rd_order_cumulant_values.append(val)

    def calc_fourth_order_cumulant(self):
        self.block_orig_4th_order_cumulant_values = []
        self.block_trans_4th_order_cumulant_values = []
        for v in self.block_vectors_orig:
            val = self.calc_cum(v, 4)
            self.block_orig_4th_order_cumulant_values.append(val)
        for v in self.block_vectors_trans:
            val = self.calc_cum(v, 4)
            self.block_trans_4th_order_cumulant_values.append(val)

    def calc_mean_of_2d_stats(self):
        self.block_means_std = []
        self.block_means_3rd_order_cumulant = []
        self.block_means_4th_order_cumulant = []
        for v in range(len(self.block_orig_std_values)):
            self.block_means_std.append(
                (self.block_orig_std_values[v] + self.block_trans_std_values[v]) / 2.0)
            self.block_means_3rd_order_cumulant.append(
                (self.block_orig_3rd_order_cumulant_values[v] + self.block_trans_3rd_order_cumulant_values[v]) / 2.0)
            self.block_means_4th_order_cumulant.append(
                (self.block_orig_4th_order_cumulant_values[v] + self.block_trans_4th_order_cumulant_values[v]) / 2.0)

    def vector_quantization(self):
        # STD VALUES
        # ----------
        min_max_scaler = MinMaxScaler((-7, 7))
        minmax_scaled = min_max_scaler.fit_transform(np.asarray(self.block_means_std).reshape(-1,1))
        quantized_std = ((ds_quantize(minmax_scaled, n=8) + 7) / 2).astype(int).ravel()  # still a numpy int array
        quantized_std_bitstring = sum(map(lambda x: BitArray(uint=x, length=3), quantized_std))

        # 3rd CUMULANT VALUES
        # ----------
        min_max_scaler = MinMaxScaler((-63, 63))
        minmax_scaled = min_max_scaler.fit_transform(np.asarray(self.block_means_3rd_order_cumulant).reshape(-1,1))
        quantized_3rd_cum = ((ds_quantize(minmax_scaled, n=64) + 63) / 2).astype(int).ravel()
        quantized_3rd_cum_bitstring = sum(map(lambda x: BitArray(uint=x, length=6), quantized_3rd_cum))

        # 4th CUMULANT VALUES
        # ----------
        min_max_scaler = MinMaxScaler((-127, 127))
        minmax_scaled = min_max_scaler.fit_transform(np.asarray(self.block_means_4th_order_cumulant).reshape(-1,1))
        quantized_4th_cum = ((ds_quantize(minmax_scaled, n=128) + 127) / 2).astype(int).ravel()
        quantized_4th_cum_bitstring = sum(map(lambda x: BitArray(uint=x, length=7), quantized_4th_cum))

        self.phash = quantized_std_bitstring + quantized_3rd_cum_bitstring + quantized_4th_cum_bitstring

    def calc_cum(self, data, n=2):
        if n > 4 or n < 1:
            raise ValueError("k-statistics only supported for 1<=n<=4")
        n = int(n)
        S = np.zeros(n + 1)
        data = np.ravel(data)
        N = len(data)
        for k in range(1, n + 1):
            S[k] = np.sum(data**k, axis=0)
        if n == 1:
            return S[1] * 1.0/N
        elif n == 2:
            return (N*S[2] - S[1]**2.0) / (N*(N - 1.0))
        elif n == 3:
            return (2*S[1]**3 - 3*N*S[1]*S[2] + N*N*S[3]) / (N*(N - 1.0)*(N - 2.0))
        elif n == 4:
            return ((-6*S[1]**4 + 12*N*S[1]**2 * S[2] - 3*N*(N-1.0)*S[2]**2 -
                     4*N*(N+1)*S[1]*S[3] + N*N*(N+1)*S[4]) /
                     (N*(N-1.0)*(N-2.0)*(N-3.0)))
        else:
            raise ValueError("Should not be here.")

    # ------

    def convert_to_grey(self):
        self.img = skimage.color.rgb2grey(self.img)

    def resize(self):
        self.img = skimage.transform.resize(self.img, self.preprocess_resize_constants)

    def gaussian_filter(self):
        self.img = skimage.filters.gaussian_filter(self.img, self.preprocess_gaussian_filter_sigma)

    def median_filter(self):
        self.img = skimage.filters.median(self.img, self.preprocess_median_filter_neighborhood)

    def hist_equalization(self):
        self.img = skimage.exposure.equalize_hist(self.img)

    # ------

    def get_result(self):
        return self.phash

    def get_img(self):
        return self.img
