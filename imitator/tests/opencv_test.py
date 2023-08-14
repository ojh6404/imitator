#!/usr/bin/env python3
#
#

import cv2
import numpy as np
from copy import deepcopy

#マウスイベント
def onMouse(event, x, y, flag, params):
    global drag, start, end

    #レフトボタンをクリックしたら
    if event == cv2.EVENT_LBUTTONDOWN:
        #ドラッグ開始座標を記録
        start = (x, y)
        #ドラッグモードON
        drag = True

    #レフトボタンを離したら
    if event == cv2.EVENT_LBUTTONUP:
        #ドラッグ終了座標を記録
        end = (x, y)
        #ドラッグモードOFF
        drag = False
        #元画像に線を書き込んで画面更新
        cv2.line(img, start, end, (0, 0, 0))
        cv2.imshow(wname, img)

    #ドラッグ中
    if drag:
        #一時表示用の画像をコピー
        img2 = deepcopy(img)
        #コピー画像に線、ドラッグ開始座標と現在座標を結ぶ
        cv2.line(img2, start, (x, y), (0, 0, 0))
        #コピー画像で画面更新
        cv2.imshow(wname, img2)

#白い画像作成
img = np.ones((300, 400, 3), 'uint8') * 255

#パラメータ初期化
drag = False
start = (0, 0)
end = (0, 0)

#準備
wname = 'aaa'
cv2.namedWindow(wname)
cv2.setMouseCallback(wname, onMouse)

#画像を立ち上げてスタート
cv2.imshow(wname, img)
cv2.waitKey()
