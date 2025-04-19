import os
import numpy as np
import cv2
import random

def main():
    sz = 512
    rad = 10
    bimg = mk_img(sz, 0)  # 0 = blue
    bmsk = mk_msk(sz, sz, rad)
    bimg[bmsk != 0] = 0

    t1img = mk_img(sz, 1)  # 1 = green
    t2img = mk_img(sz, 2)  # 2 = red
    t1msk = mk_msk(sz, sz, rad)
    t2msk = mk_msk(sz, sz, rad)

    timgs = [t1img, t2img]
    tmsks = [t1msk, t2msk]

    for timg, tmsk in zip(timgs, tmsks):
        bimg, bmsk = abt(bimg, bmsk, timg, tmsk)

    save_img(['abt_img.jpg', 'abt_msk.png'], [bimg, bmsk])


def abt(bimg, bmsk, timg, tmsk):
    tries = 10
    for _ in range(tries):
        shx, shy = rnd_sh(tmsk)
        shmsk = sh_msk(tmsk, shy, shx)

        if not chk_ovr(bmsk, shmsk):
            bimg, bmsk = mk_abt(bmsk, bimg, timg, tmsk, shmsk)
            return bimg, bmsk

    print('Overlap: placement failed')
    return bimg, bmsk


def rnd_sh(msk):
    y, x = np.where(msk > 0)
    shy = np.random.randint(-np.min(y), msk.shape[0] - np.max(y))
    shx = np.random.randint(-np.min(x), msk.shape[1] - np.max(x))
    return shy, shx


def sh_msk(msk, shy, shx):
    y, x = np.where(msk > 0)
    sy = y + shy
    sx = x + shx
    shmsk = np.zeros_like(msk)
    shmsk[sy, sx] = msk[y, x]
    return shmsk


def chk_ovr(bmsk, shmsk):
    return np.any(np.logical_and(bmsk != 0, shmsk != 0))


def mk_abt(bmsk, bimg, timg, tmsk, shmsk):
    tobj = timg.copy()
    tobj[tmsk == 0] = 0
    oimg = bimg.copy()
    oimg[shmsk != 0] = tobj[tmsk != 0]
    omsk = np.where(shmsk != 0, shmsk, bmsk)
    return oimg, omsk


def mk_img(sz, col):
    img = np.zeros((sz, sz, 3), np.uint8)
    img[:, :, col] = 255
    return img


def mk_msk(h, w, rad, ctr=None):
    if ctr is None:
        ctr = (int(w / 2), int(h / 2))
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((X - ctr[0])**2 + (Y - ctr[1])**2)
    msk = dist <= rad
    return np.uint8(msk * 255)


def save_img(fns, imgs):
    for fn, img in zip(fns, imgs):
        cv2.imwrite(fn, img)


if __name__ == '__main__':
    main()