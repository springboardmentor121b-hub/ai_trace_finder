import numpy as np
import tensorflow as tf
import cv2
from skimage.feature import local_binary_pattern as sk_lbp
from scipy.fft import fft2, fftshift
from scipy import ndimage

# --------------------------------------------------
# GPU availability
# --------------------------------------------------
gpus = tf.config.list_physical_devices('GPU')
USE_GPU = len(gpus) > 0
print(f"Hybrid utils: Use GPU? {USE_GPU}")

# --------------------------------------------------
# Correlation (CPU)
# --------------------------------------------------
def corr2d(a, b):
    a = a.astype(np.float32).ravel()
    b = b.astype(np.float32).ravel()
    a -= a.mean()
    b -= b.mean()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float((a @ b) / denom) if denom > 0 else 0.0

# --------------------------------------------------
# Batch Correlation (GPU-safe)
# --------------------------------------------------
def batch_corr_gpu(residuals, fingerprints, fp_keys):
    """
    residuals: np.ndarray (N, H, W)
    fingerprints: dict[str -> (H, W)]
    fp_keys: ordered scanner list
    """
    if residuals is None or len(residuals) == 0:
        return np.empty((0, len(fp_keys)), dtype=np.float32)

    # ---- Fingerprint matrix (D, K)
    fp_mat = []
    for k in fp_keys:
        fp = fingerprints[k].astype(np.float32).ravel()
        fp -= fp.mean()
        n = np.linalg.norm(fp)
        if n > 0:
            fp /= n
        fp_mat.append(fp)

    fp_mat = np.stack(fp_mat, axis=1)  # (D, K)

    # ---- Residual matrix (N, D)
    res_mat = []
    for r in residuals:
        r = r.astype(np.float32).ravel()
        r -= r.mean()
        n = np.linalg.norm(r)
        if n > 0:
            r /= n
        res_mat.append(r)

    res_mat = np.stack(res_mat, axis=0)  # (N, D)

    # ---- Matmul
    device = '/GPU:0' if USE_GPU else '/CPU:0'
    with tf.device(device):
        A = tf.convert_to_tensor(res_mat, tf.float32)
        B = tf.convert_to_tensor(fp_mat, tf.float32)
        C = tf.matmul(A, B)

    return C.numpy()

# --------------------------------------------------
# FFT radial energy
# --------------------------------------------------
def fft_radial_energy(img, K=6):
    f = fftshift(fft2(img))
    mag = np.abs(f)
    h, w = mag.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.ogrid[:h, :w]
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    rmax = r.max() + 1e-6
    bins = np.linspace(0, rmax, K + 1)
    return [float(np.mean(mag[(r >= bins[i]) & (r < bins[i+1])])) for i in range(K)]

# --------------------------------------------------
# LBP histogram
# --------------------------------------------------
def lbp_hist_safe(img, P=8, R=1.0):
    rng = np.ptp(img)
    g = np.zeros_like(img) if rng < 1e-12 else (img - img.min()) / (rng + 1e-8)
    g8 = (g * 255).astype(np.uint8)
    codes = sk_lbp(g8, P=P, R=R, method="uniform")
    bins = P + 2
    hist, _ = np.histogram(codes, bins=bins, range=(0, bins), density=True)
    return hist.astype(np.float32).tolist()

# --------------------------------------------------
# Enhanced features
# --------------------------------------------------
def extract_enhanced_features(residual):
    f = fftshift(fft2(residual))
    mag = np.abs(f)

    h, w = mag.shape
    ch, cw = h // 2, w // 2

    low = np.mean(mag[ch-20:ch+20, cw-20:cw+20])
    mid = np.mean(mag[ch-60:ch+60, cw-60:cw+60]) - low
    high = np.mean(mag) - np.mean(mag[ch-60:ch+60, cw-60:cw+60])

    lbp = sk_lbp(residual, P=24, R=3, method='uniform')
    lbp_hist, _ = np.histogram(lbp, bins=26, range=(0, 26), density=True)

    gx = ndimage.sobel(residual, axis=1)
    gy = ndimage.sobel(residual, axis=0)
    gmag = np.sqrt(gx**2 + gy**2)

    texture = [
        float(np.std(residual)),
        float(np.mean(np.abs(residual))),
        float(np.std(gmag)),
        float(np.mean(gmag))
    ]

    return [float(low), float(mid), float(high)] + lbp_hist.tolist() + texture

# --------------------------------------------------
# Image helpers
# --------------------------------------------------
def to_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img

def resize_to(img, size=(256, 256)):
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)

def normalize_img(img):
    return img.astype(np.float32) / 255.0

# --------------------------------------------------
# Batch preprocessing (GPU)
# --------------------------------------------------
def process_batch_gpu(file_paths):
    imgs = []

    for fpath in file_paths:
        img = cv2.imread(fpath, cv2.IMREAD_UNCHANGED)
        if img is None:
            continue
        img = to_gray(img)
        img = resize_to(img)
        img = normalize_img(img)
        imgs.append(img)

    if len(imgs) == 0:
        return []

    batch = np.expand_dims(np.array(imgs, dtype=np.float32), -1)

    device = '/GPU:0' if USE_GPU else '/CPU:0'
    with tf.device(device):
        x = tf.convert_to_tensor(batch)
        pooled = tf.nn.avg_pool2d(x, 2, 2, 'VALID')
        den = tf.image.resize(pooled, (256, 256), method='nearest')
        res = x - den

    return [r.squeeze() for r in res.numpy()]
