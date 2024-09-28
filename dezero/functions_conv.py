import numpy as np
from dezero import cuda
from dezero.core import Function, as_variable
from dezero.utils import pair, get_conv_outsize, get_deconv_outsize
from dezero.functions import linear, broadcast_to


# =============================================================================
# [simple version] conv2d_simple / pooling_simple
# =============================================================================
def conv2D_simple(x, W, b=None, stride=1, pad=0):
    # example: for one image data (N = 1)
    # oc: # of filters to apply = output channels
    # img (1, c, h, w) @ filter (oc, c, kh, kw)
    # -> output (1, oc, oh, ow) + bias (oc, 1, 1) -> output (1, oc, oh, ow)

    x, W = as_variable(x), as_variable(W)
    # input image x : (batch size N, # of channels, height, width)
    N, c, h, w = x.shape
    oc, c, kh, kw = W.shape  # shape of filter
    sh, sw = pair(stride)
    ph, pw = pair(pad)
    oh = get_conv_outsize(h, kh, sh, ph)
    ow = get_conv_outsize(w, kw, sw, pw)

    # col : (N x oh x ow, c x kh x kw) - transformed input image
    col = im2col(x, (kh, kw), stride, pad, to_matrix=True)
    # w: (oc, c, kh, kw) -> (c x kh x kw, oc)
    W = W.reshape(oc, -1).transpose()
    t = linear(col, W, b)  # t = col*W + b where b.T.shape = (oc, 1, 1)
    # t: (N x oh x ow, oc) -> (N, oc, oh, ow)
    y = t.reshape(N, oh, ow, oc).transpose(0, 3, 1, 2)
    return y


def pooling_simple(x, kernel_size, stride=1, pad=0):
    x = as_variable(x)

    N, C, H, W = x.shape
    KH, KW = pair(kernel_size)
    SH, SW = pair(stride)
    PH, PW = pair(pad)
    OH = get_conv_outsize(H, KH, SH, PH)
    OW = get_conv_outsize(W, KW, SW, PW)

    # col: (N x OH x OW, C x KH x KW) -> reshape (N x C x OH x OW, KH x KW)
    col = im2col(x, kernel_size, stride, pad, to_matrix=True)
    col = col.reshape(-1, KH * KW)
    # max pooling
    y = col.max(axis=1)
    y = y.reshape(N, OH, OW, C).transpose(0, 3, 1, 2)
    return y


# =============================================================================
#  conv2d / deconv2d
# =============================================================================
class Conv2D(Function):
    def __init__(self, stride, pad):
        super().__init__()
        self.stride = pair(stride)
        self.pad = pair(pad)

    def forward(self, x, W, b):
        xp = cuda.get_array_module(x)
        # x.shape = (N, c, h, w)
        # W.shape = (oc, c, kh, kw)
        kh, kw = W.shape[2:]
        # col.shape = (N, C, KH, KW, OH, OW)
        col = im2col_array(x, (kh, kw), self.stride, self.pad, to_matrix=False)
        y = xp.tensordot(col, W, ((1, 2, 3), (1, 2, 3)))  # (N, OH, OW, OC)

        if b is not None:
            y += b  # b : (OC, )

        # final shape of y: (N, OC, OH, OW)
        y = xp.rollaxis(y, 3, 1)  # equal to y = np.transpose(y, (0, 3, 1, 2))
        return y

    def backward(self, gy):
        x, W, b = self.inputs
        # calculate gx
        gx = deconv2d(
            gy,
            W,
            b=None,
            stride=self.stride,
            pad=self.pad,
            outsize=(x.shape[2], x.shape[3]),
        )
        # calculate gW
        gW = Conv2DGradW(self)(x, gy)
        # gb
        gb = None
        if b.data is not None:
            gb = gy.sum(axis=(0, 2, 3))  # (OC,)
        return gx, gW, gb


def conv2d(x, W, b=None, stride=1, pad=0):
    return Conv2D(stride, pad)(x, W, b)


class Deconv2D(Function):
    def __init__(self, stride, pad, outsize):
        super().__init__()
        self.stride = pair(stride)
        self.pad = pair(pad)
        self.outsize = outsize

    def forward(self, x, W, b):
        xp = cuda.get_array_module(x)

        sh, sw = self.stride
        ph, pw = self.pad
        c, oc, kh, kw = W.shape
        N, c, h, w = x.shape

        if self.outsize is None:
            out_h = get_deconv_outsize(h, kh, sh, ph)
            out_w = get_deconv_outsize(w, kw, sw, pw)
        else:
            out_h, out_w = pair(self.outsize)
        img_shape = (N, oc, out_h, out_w)

        gcol = xp.tensordot(W, x, (0, 1))
        gcol = xp.rollaxis(gcol, 3)
        y = col2im_array(
            gcol, img_shape, (kh, kw), self.stride, self.pad, to_matrix=False
        )
        # b, k, h, w
        if b is not None:
            self.no_bias = True
            y += b.reshape((1, b.size, 1, 1))
        return y

    def backward(self, gy):
        x, W, b = self.inputs

        # ==== gx ====
        gx = conv2d(gy, W, b=None, stride=self.stride, pad=self.pad)
        # ==== gW ====
        f = Conv2DGradW(self)
        gW = f(gy, x)
        # ==== gb ====
        gb = None
        if b.data is not None:
            gb = gy.sum(axis=(0, 2, 3))
        return gx, gW, gb


def deconv2d(x, W, b=None, stride=1, pad=0, outsize=None):
    return Deconv2D(stride, pad, outsize)(x, W, b)


class Conv2DGradW(Function):
    def __init__(self, conv2d):
        # x, W, b = conv2d.inputs
        # W.shape = (c, oc, kh, kw)
        W = conv2d.inputs[1]
        kh, kw = W.shape[2:]
        self.kernel_size = (kh, kw)
        self.stride = conv2d.stride
        self.pad = conv2d.pad

    def forward(self, x, gy):
        xp = cuda.get_array_module(x)
        # x - > col: col.shape = (N, C, KH, KW, OH, OW)
        col = im2col_array(x, self.kernel_size, self.stride, self.pad, to_matrix=False)
        # gy: (N, OC, OH, OW)
        gW = xp.tensordot(gy, col, ((0, 2, 3), (0, 4, 5)))
        return gW

    def backward(self, gys):
        x, gy = self.inputs
        (gW,) = self.outputs
        xh, xw = x.shape[2:]
        gx = deconv2d(gy, gW, stride=self.stride, pad=self.pad, outsize=(xh, xw))
        ggy = conv2d(x, gW, stride=self.stride, pad=self.pad)
        return gx, ggy


# =============================================================================
#  pooling(max-pooling) / average_pooling
# =============================================================================
class Pooling(Function):
    def __init__(self, kernel_size, stride=1, pad=0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad = pad

    def forward(self, x):
        col = im2col_array(x, self.kernel_size, self.stride, self.pad, to_matrix=False)
        N, C, KH, KW, OH, OW = col.shape
        col = col.reshape(N, C, KH * KW, OH, OW)
        self.idx = col.argmax(axis=2)
        y = col.max(axis=2)
        return y

    def backward(self, gy):
        return Pooling2DGrad(self)(gy)


class Pooling2DGrad(Function):
    def __init__(self, max_pool2d):
        self.max_pool2d = max_pool2d
        self.kernel_size = max_pool2d.kernel_size
        self.stride = max_pool2d.stride
        self.pad = max_pool2d.pad
        self.input_shape = max_pool2d.inputs[0].shape
        self.dtype = max_pool2d.inputs[0].dtype
        self.idx = max_pool2d.idx

    def forward(self, gy):
        xp = cuda.get_array_module(gy)

        N, C, OH, OW = gy.shape
        N, C, H, W = self.input_shape
        KH, KW = pair(self.kernel_size)

        gcol = xp.zeros((N * C * OH * OW * KH * KW), dtype=self.dtype)
        idx = self.idx.ravel() + xp.arange(0, self.idx.size * KH * KW, KH * KW)
        gcol = gcol.reshape(N, C, OH, OW, KH, KW)
        gcol = xp.swapaxes(gcol, 2, 4)
        gcol = xp.swapaxes(gcol, 3, 5)  # (N, C, KH, KW, OH, OW)
        gx = col2im_array(
            gcol, (N, C, H, W), self.kernel_size, self.stride, self.pad, to_matrix=False
        )
        return gx

    def backward(self, ggx):
        f = Pooling2DWithIndexes(self.max_pool2d)
        return f(ggx)


class Pooling2DWithIndexes(Function):
    # apply the gradients to the correct positions using the saved indices from max pooling.
    def __init__(self, mpool2d):
        self.kernel_size = mpool2d.kernel_size
        self.stride = mpool2d.stride
        self.pad = mpool2d.pad
        self.input_shpae = mpool2d.inputs[0].shape
        self.dtype = mpool2d.inputs[0].dtype
        self.indexes = mpool2d.indexes

    def forward(self, x):
        col = im2col_array(x, self.kernel_size, self.stride, self.pad, to_matrix=False)
        N, C, KH, KW, OH, OW = col.shape
        col = col.reshape(N, C, KH * KW, OH, OW)
        col = col.transpose(0, 1, 3, 4, 2).reshape(-1, KH * KW)
        indexes = self.indexes.ravel()
        col = col[np.arange(len(indexes)), indexes]
        return col.reshape(N, C, OH, OW)


def pooling(x, kernel_size, stride=1, pad=0):
    return Pooling(kernel_size, stride, pad)(x)


class AveragePooling(Function):
    def __init__(self, kernel_size, stride=1, pad=0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad = pad
        self.input_size = None

    def forward(self, x):
        self.input_shape = x.shape
        col = im2col_array(x, self.kernel_size, self.stride, self.pad, to_matrix=False)
        y = col.mean(axis=(2, 3))
        return y

    def backward(self, gy):
        N, C, OH, OW = gy.shape
        KW, KH = pair(self.kernel_size)
        gy /= KH * KW
        gcol = broadcast_to(gy.reshape(-1), (KH, KW, N * C * OH * OW))
        gcol = gcol.reshape(KH, KW, N, C, OH, OW).transpose(2, 3, 0, 1, 4, 5)
        gx = col2im(
            gcol,
            self.input_shape,
            self.kernel_size,
            self.stride,
            self.pad,
            to_matrix=False,
        )
        return gx


def average_pooling(x, kernel_size, stride=1, pad=0):
    return AveragePooling(kernel_size, stride, pad)(x)


# =============================================================================
#  im2col / col2im
# =============================================================================
class Im2Col(Function):
    def __init__(self, kernel_size, stride, pad, to_matrix):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad = pad
        self.to_matrix = to_matrix

    def forward(self, x):
        self.input_shape = x.shape
        y = im2col_array(x, self.kernel_size, self.stride, self.pad, self.to_matrix)
        return y

    def backward(self, gy):
        gx = col2im(
            gy,
            self.input_shape,
            self.kernel_size,
            self.stride,
            self.pad,
            self.to_matrix,
        )
        return gx


def im2col(x, kernel_size, stride=1, pad=0, to_matrix=True):
    """Extract patches from an image based on the filter.

    Args:
        x (`dezero.Variable` or `ndarray`): Input variable of shape
            `(N, C, H, W)`
        kernel_size (int or (int, int)): Size of kernel.
        stride (int or (int, int)): Stride of kernel.
        pad (int or (int, int)): Spatial padding width for input arrays.
        to_matrix (bool): If True the `col` will be reshaped to 2d array whose
            shape is `(N*OH*OW, C*KH*KW)`

    Returns:
        `dezero.Variable`: Output variable. If the `to_matrix` is False, the
            output shape is `(N, C, KH, KW, OH, OW)`, otherwise
            `(N*OH*OW, C*KH*KW)`.

    Notation:
    - `N` is the batch size.
    - `C` is the number of the input channels.
    - `H` and `W` are the height and width of the input image, respectively.
    - `KH` and `KW` are the height and width of the filters, respectively.
    - `SH` and `SW` are the strides of the filter.
    - `PH` and `PW` are the spatial padding sizes.
    - `OH` and `OW` are the the height and width of the output, respectively.
    """
    y = Im2Col(kernel_size, stride, pad, to_matrix)(x)
    return y


class Col2Im(Function):
    def __init__(self, input_shape, kernel_size, stride, pad, to_matrix):
        super().__init__()
        self.input_shape = input_shape
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad = pad
        self.to_matrix = to_matrix

    def forward(self, x):
        y = col2im_array(
            x, self.input_shape, self.kernel_size, self.stride, self.pad, self.to_matrix
        )
        return y

    def backward(self, gy):
        gx = im2col(gy, self.kernel_size, self.stride, self.pad, self.to_matrix)
        return gx


def col2im(x, input_shape, kernel_size, stride=1, pad=0, to_matrix=True):
    return Col2Im(input_shape, kernel_size, stride, pad, to_matrix)(x)


# =============================================================================
#  numpy im2col
# =============================================================================


def im2col_array(img, kernel_size, stride, pad, to_matrix=True):
    N, C, H, W = img.shape
    KH, KW = pair(kernel_size)
    SH, SW = pair(stride)
    PH, PW = pair(pad)
    OH = get_conv_outsize(H, KH, SH, PH)
    OW = get_conv_outsize(W, KW, SW, PW)

    xp = cuda.get_array_module(img)
    if xp != np:
        col = _im2col_gpu(img, kernel_size, stride, pad)
    else:
        # np.pad(array - image in this case, pad_width, mode='constant', constant_values = 0)
        # pad_width = ((0, 0) for N, (0, 0) for C, (PH, PH + SH - 1) for H, (PW, PW + SW - 1) for W)
        # the firt two (0,0) : no padding for batch dimension (N) and channel dimension (C)
        # (PH, PH + SH - 1) adds padding of size PH to the top and PH + SH - 1 to the bottom of the image
        # (PW, PW + SW - 1) adds padding to the left and right of the image width.
        # why PH + SH -1 ? When using a stride greater than 1, the kernel moves in larger steps (e.g., if stride = 2, it moves by 2 pixels at a time), so the padding at the bottom may need to be increased slightly to ensure the kernel still covers the last pixel(s).
        # This adjustment prevents the kernel from missing the last few pixels along the bottom when performing convolution. Adding SH - 1 to the bottom padding compensates for the effect of the stride.
        img = np.pad(
            img,
            ((0, 0), (0, 0), (PH, PH + SH - 1), (PW, PW + SW - 1)),
            mode="constant",
            constant_values=0,
        )

        # col : empty array - will store the im2col-transformed output
        col = np.ndarray((N, C, KH, KW, OH, OW), dtype=img.dtype)
        # extract image patches from input image at different strides and store them in col.
        for j in range(KH):
            j_lim = j + SH * OH
            for i in range(KW):
                i_lim = i + SW * OW
                # fills the col array by extracting patches from img.
                # Each patch corresponds to a segment of the image that the convolution kernel will operate on.
                col[:, :, j, i, :, :] = img[:, :, j:j_lim:SH, i:i_lim:SW]

    if to_matrix:
        # reshape col from (N (0), C(1), KH(2), KW(3), OH(4), OW(5)) to (N, OH x OW, C x KH x KW) for matrix operation
        col = col.transpose((0, 4, 5, 1, 2, 3)).reshape(N * OH * OW, -1)
    return col


def col2im_array(col, img_shape, kernel_size, stride, pad, to_matrix=True):
    N, C, H, W = img_shape
    KH, KW = pair(kernel_size)
    SH, SW = pair(stride)
    PH, PW = pair(pad)
    OH = get_conv_outsize(H, KH, SH, PH)
    OW = get_conv_outsize(W, KW, SW, PW)

    if to_matrix:
        col = col.reshape(N, OH, OW, C, KH, KW).transpose((0, 3, 4, 5, 1, 2))
    xp = cuda.get_array_module(col)
    if xp != np:
        img = _col2im_gpu(col, SH, SW, PH, PW, H, W)
        return img
    else:
        # height of the original image, plus the padding on both sides (2 * PH), plus the extra offset from the stride (SH - 1).
        img = np.zeros(
            (N, C, H + 2 * PH + SH - 1, W + 2 * PW + SW - 1), dtype=col.dtype
        )
        # padded image (img) will be filled with the reconstructed values by adding back patches (stored in col) that were extracted during im2col
        for j in range(KH):
            j_lim = j + SH * OH
            for i in range(KW):
                i_lim = i + SW * OH
                img[:, :, j:j_lim:SH, i:i_lim:SW] += col[:, :, j, i, :, :]
        # crops out the extra padding that was added at the start. The final image is of shape (N, C, H, W), which removes the padding from both the height and width dimensions.
        return img[:, :, PH : H + PH, PW : W + PW]


def _im2col_gpu(img, kernel_size, stride, pad):
    """im2col function for GPU.
    This code is ported from Chainer:
    https://github.com/chainer/chainer/blob/v6.4.0/chainer/utils/conv.py
    """
    n, c, h, w = img.shape
    kh, kw = pair(kernel_size)
    sy, sx = pair(stride)
    ph, pw = pair(pad)
    out_h = get_conv_outsize(h, kh, sy, ph)
    out_w = get_conv_outsize(w, kw, sx, pw)
    dy, dx = 1, 1
    col = cuda.cupy.empty((n, c, kh, kw, out_h, out_w), dtype=img.dtype)

    cuda.cupy.ElementwiseKernel(
        "raw T img, int32 h, int32 w, int32 out_h, int32 out_w,"
        "int32 kh, int32 kw, int32 sy, int32 sx, int32 ph, int32 pw,"
        "int32 dy, int32 dx",
        "T col",
        """
           int c0 = i / (kh * kw * out_h * out_w);
           int ky = i / (kw * out_h * out_w) % kh;
           int kx = i / (out_h * out_w) % kw;
           int out_y = i / out_w % out_h;
           int out_x = i % out_w;
           int in_y = ky * dy + out_y * sy - ph;
           int in_x = kx * dx + out_x * sx - pw;
           if (in_y >= 0 && in_y < h && in_x >= 0 && in_x < w) {
             col = img[in_x + w * (in_y + h * c0)];
           } else {
             col = 0;
           }
        """,
        "im2col",
    )(img.reduced_view(), h, w, out_h, out_w, kh, kw, sy, sx, ph, pw, dy, dx, col)

    return col


def _col2im_gpu(col, sy, sx, ph, pw, h, w):
    """col2im function for GPU.
    This code is ported from Chainer:
    https://github.com/chainer/chainer/blob/v6.4.0/chainer/utils/conv.py
    """
    n, c, kh, kw, out_h, out_w = col.shape
    dx, dy = 1, 1
    img = cuda.cupy.empty((n, c, h, w), dtype=col.dtype)

    cuda.cupy.ElementwiseKernel(
        "raw T col, int32 h, int32 w, int32 out_h, int32 out_w,"
        "int32 kh, int32 kw, int32 sy, int32 sx, int32 ph, int32 pw,"
        "int32 dx, int32 dy",
        "T img",
        """
           int c0 = i / (h * w);
           int y  = i / w % h;
           int x  = i % w;
           T val = 0;
           for (int ky = 0; ky < kh; ++ky) {
             int out_y = (y + ph - ky * dy);
             if (0 > out_y || out_y >= out_h * sy) continue;
             if (out_y % sy != 0) continue;
             out_y /= sy;
             for (int kx = 0; kx < kw; ++kx) {
               int out_x = (x + pw - kx * dx);
               if (0 > out_x || out_x >= out_w * sx) continue;
               if (out_x % sx != 0) continue;
               out_x /= sx;
               int k = out_y + out_h * (kx + kw * (ky + kh * c0));
               val = val + col[out_x + out_w * k];
             }
           }
           img = val;
        """,
        "col2im",
    )(col.reduced_view(), h, w, out_h, out_w, kh, kw, sy, sx, ph, pw, dx, dy, img)
    return img
