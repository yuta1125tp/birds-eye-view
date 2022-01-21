#%%
cd ..
#%%
from pathlib import Path
import cv2
from easydict import EasyDict as edict
import numpy as np
import sys
import os
import pickle as pkl
from matplotlib import pyplot as plt
import math
sys.path.insert(0, "scripts")

import predict_horizon_vpz_homography as vpz_tool
# %%
imgA_stem = "scene0713_00_frame-001320"
imgB_stem = "scene0713_00_frame-002025"

pkl_file = Path("output")/f"{imgA_stem}.inception-v4.pkl"
pklobjA = edict(pkl.loads(pkl_file.read_bytes()))
pkl_file = Path("output")/f"{imgB_stem}.inception-v4.pkl"
pklobjB = edict(pkl.loads(pkl_file.read_bytes()))

# %%
plt.imshow(pklobjA.img_cv)
plt.show()
plt.imshow(pklobjB.img_cv)
plt.show()

# # %%
# ax = vpz_tool.plot_scaled_horizonvector_vpz_picture(
#     pklobjA.img_cv,
#     pklobjA.estimated_input_points,
#     net_dims=(pklobjA.net_width, pklobjA.net_height),
#     color="go",
#     show_vz=True,
#     verbose=True,
# )
# plt.show()

# %%
def warp_func(
    img_cv,
    estimated_input_points,
    net_width,
    net_height,
    orig_width,
    orig_height,
    verbose=False,
):
    (
        fx,
        fy,
        roll_from_horizon,
        my_tilt,
    ) = vpz_tool.get_intrinisic_extrinsic_params_from_horizonvector_vpz(
        img_dims=(orig_width, orig_height),
        horizonvector_vpz=estimated_input_points,
        net_dims=(net_width, net_height),
        verbose=verbose,
    )
    
    K3x3 = np.array(
        [
            [fx, 0, orig_width / 2],
            [0, fy, orig_height / 2],
            [0, 0, 1],
        ]
    )

    (
        overhead_hmatrix,
        est_range_u,
        est_range_v,
    ) = vpz_tool.get_overhead_hmatrix_from_4cameraparams(
        fx=fx,
        fy=fy,
        my_tilt=my_tilt,
        my_roll=-math.radians(roll_from_horizon),
        img_dims=(orig_width, orig_height),
        verbose=verbose,
    )

    scaled_overhead_hmatrix, target_dim = vpz_tool.get_scaled_homography(
        overhead_hmatrix, 1080 * 2, est_range_u, est_range_v
    )

    warped = cv2.warpPerspective(
        img_cv, scaled_overhead_hmatrix, dsize=target_dim, flags=cv2.INTER_CUBIC
    )
    plt.imshow(warped[:,:,::-1])
    plt.show()
    return scaled_overhead_hmatrix, K3x3

# %%
homoA, KA = warp_func(
    **pklobjA,
    orig_width=pklobjA.img_cv.shape[1],
    orig_height=pklobjA.img_cv.shape[0],
)
homoB, KB = warp_func(
    **pklobjB,
    orig_width=pklobjB.img_cv.shape[1],
    orig_height=pklobjB.img_cv.shape[0],
)
# %%

npz_path = Path("..\\SuperGluePretrainedNetwork\\dump_match_pairs\\scene0713_00_frame-001320_scene0713_00_frame-002025_matches.npz")
npz_path.exists()
# %%
npzobj = edict(np.load(str(npz_path)))
print(npzobj)
#%%
def estimate_pose(kpts0, kpts1, K0, K1, thresh, conf=0.99999):
    if len(kpts0) < 5:
        return None

    f_mean = np.mean([K0[0, 0], K1[1, 1], K0[0, 0], K1[1, 1]])
    norm_thresh = thresh / f_mean

    kpts0 = (kpts0 - K0[[0, 1], [2, 2]][None]) / K0[[0, 1], [0, 1]][None]
    kpts1 = (kpts1 - K1[[0, 1], [2, 2]][None]) / K1[[0, 1], [0, 1]][None]

    E, mask = cv2.findEssentialMat(
        kpts0, kpts1, np.eye(3), threshold=norm_thresh, prob=conf,
        method=cv2.RANSAC)

    assert E is not None

    best_num_inliers = 0
    ret = None
    for _E in np.split(E, len(E) / 3):
        n, R, t, _ = cv2.recoverPose(
            _E, kpts0, kpts1, np.eye(3), 1e9, mask=mask)
        if n > best_num_inliers:
            best_num_inliers = n
            ret = (R, t[:, 0], mask.ravel() > 0)
    return ret


# %%
matches = npzobj.matches
kpts0 = npzobj.keypoints0
kpts1 = npzobj.keypoints1
conf = npzobj.match_confidence

valid = matches > -1
mkpts0 = kpts0[valid]
mkpts1 = kpts1[matches[valid]]
mconf = conf[valid]

K0 = KA  # np.array(pair[4:13]).astype(float).reshape(3, 3)
K1 = KB  # np.array(pair[13:22]).astype(float).reshape(3, 3)
# T_0to1 = np.array(pair[22:]).astype(float).reshape(4, 4)

# # Scale the intrinsics to resized image.
# K0 = scale_intrinsics(K0, scales0)
# K1 = scale_intrinsics(K1, scales1)

# # Update the intrinsics + extrinsics if EXIF rotation was found.
# if rot0 != 0 or rot1 != 0:
#     cam0_T_w = np.eye(4)
#     cam1_T_w = T_0to1
#     if rot0 != 0:
#         K0 = rotate_intrinsics(K0, image0.shape, rot0)
#         cam0_T_w = rotate_pose_inplane(cam0_T_w, rot0)
#     if rot1 != 0:
#         K1 = rotate_intrinsics(K1, image1.shape, rot1)
#         cam1_T_w = rotate_pose_inplane(cam1_T_w, rot1)
#     cam1_T_cam0 = cam1_T_w @ np.linalg.inv(cam0_T_w)
#     T_0to1 = cam1_T_cam0

# epi_errs = compute_epipolar_error(mkpts0, mkpts1, T_0to1, K0, K1)
# correct = epi_errs < 5e-4
# num_correct = np.sum(correct)
# precision = np.mean(correct) if len(correct) > 0 else 0
# matching_score = num_correct / len(kpts0) if len(kpts0) > 0 else 0

thresh = 1.  # In pixels relative to resized image size.
ret = estimate_pose(mkpts0, mkpts1, K0, K1, thresh)
if ret is None:
    err_t, err_R = np.inf, np.inf
else:
    R, t, inliers = ret

# %%
print(R)
print(t)
# %%

imgA = pklobjA.img_cv
imgB = pklobjB.img_cv

# %%

M, mask = cv2.findHomography(mkpts0, mkpts1, cv2.RANSAC, 5.0)
matchesMask = mask.ravel().tolist()

h,w = imgA.shape[:2]
pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
dst = cv2.perspectiveTransform(pts,M)

import copy
dst = cv2.polylines(copy.deepcopy(imgB),[np.int32(dst)], True,255,3, cv2.LINE_AA)


# %%
plt.imshow(imgA)
plt.show()
plt.imshow(imgB)
plt.show()
plt.imshow(dst)
plt.show()
# %%
plt.imshow(
cv2.warpPerspective(
    cv2.resize(imgA, (640,480)), 
    #imgA,
    M, dsize=(2048,2048))
)
plt.show()
plt.imshow(
cv2.warpPerspective(
    cv2.resize(imgB, (640,480)), 
    #imgB,
    np.eye(3), dsize=(2048,2048))
)
plt.show()
#%%
warpedA=cv2.warpPerspective(
    cv2.resize(imgA, (640,480)), 
    #imgA,
    M, dsize=(2048,2048))
warpedB=cv2.warpPerspective(
    cv2.resize(imgB, (640,480)), 
    #imgA,
    np.eye(3), dsize=(2048,2048))
plt.imshow(((warpedA/255.+warpedB/255.)/2.)[:700,:700])
plt.show()

# %%
img = cv2.imread(str(npz_path.with_suffix(".png")))
plt.imshow(img)
plt.show()


# %%

# --- VISUALIZATION ---
import matplotlib

def plot_image_pair(imgs, dpi=100, size=6, pad=.5):
    n = len(imgs)
    assert n == 2, 'number of images must be two'
    figsize = (size*n, size*3/4) if size is not None else None
    _, ax = plt.subplots(1, n, figsize=figsize, dpi=dpi)
    for i in range(n):
        ax[i].imshow(imgs[i], cmap=plt.get_cmap('gray'), vmin=0, vmax=255)
        ax[i].get_yaxis().set_ticks([])
        ax[i].get_xaxis().set_ticks([])
        for spine in ax[i].spines.values():  # remove frame
            spine.set_visible(False)
    plt.tight_layout(pad=pad)


def plot_keypoints(kpts0, kpts1, color='w', ps=2):
    ax = plt.gcf().axes
    ax[0].scatter(kpts0[:, 0], kpts0[:, 1], c=color, s=ps)
    ax[1].scatter(kpts1[:, 0], kpts1[:, 1], c=color, s=ps)


def plot_matches(kpts0, kpts1, color, lw=1.5, ps=4):
    fig = plt.gcf()
    ax = fig.axes
    fig.canvas.draw()

    transFigure = fig.transFigure.inverted()
    fkpts0 = transFigure.transform(ax[0].transData.transform(kpts0))
    fkpts1 = transFigure.transform(ax[1].transData.transform(kpts1))

    fig.lines = [matplotlib.lines.Line2D(
        (fkpts0[i, 0], fkpts1[i, 0]), (fkpts0[i, 1], fkpts1[i, 1]), zorder=1,
        transform=fig.transFigure, c=color[i], linewidth=lw)
                 for i in range(len(kpts0))]
    ax[0].scatter(kpts0[:, 0], kpts0[:, 1], c=color, s=ps)
    ax[1].scatter(kpts1[:, 0], kpts1[:, 1], c=color, s=ps)


def make_matching_plot(image0, image1, kpts0, kpts1, mkpts0, mkpts1,
                       color, text, path, show_keypoints=False,
                       fast_viz=False, opencv_display=False,
                       opencv_title='matches', small_text=[]):

    if fast_viz:
        make_matching_plot_fast(image0, image1, kpts0, kpts1, mkpts0, mkpts1,
                                color, text, path, show_keypoints, 10,
                                opencv_display, opencv_title, small_text)
        return

    plot_image_pair([image0, image1])
    if show_keypoints:
        plot_keypoints(kpts0, kpts1, color='k', ps=4)
        plot_keypoints(kpts0, kpts1, color='w', ps=2)
    plot_matches(mkpts0, mkpts1, color)

    fig = plt.gcf()
    txt_color = 'k' if image0[:100, :150].mean() > 200 else 'w'
    fig.text(
        0.01, 0.99, '\n'.join(text), transform=fig.axes[0].transAxes,
        fontsize=15, va='top', ha='left', color=txt_color)

    txt_color = 'k' if image0[-100:, :150].mean() > 200 else 'w'
    fig.text(
        0.01, 0.01, '\n'.join(small_text), transform=fig.axes[0].transAxes,
        fontsize=5, va='bottom', ha='left', color=txt_color)

    plt.savefig(str(path), bbox_inches='tight', pad_inches=0)
    plt.close()


def make_matching_plot_fast(image0, image1, kpts0, kpts1, mkpts0,
                            mkpts1, color, text, path=None,
                            show_keypoints=False, margin=10,
                            opencv_display=False, opencv_title='',
                            small_text=[]):
    H0, W0 = image0.shape
    H1, W1 = image1.shape
    H, W = max(H0, H1), W0 + W1 + margin

    out = 255*np.ones((H, W), np.uint8)
    out[:H0, :W0] = image0
    out[:H1, W0+margin:] = image1
    out = np.stack([out]*3, -1)

    if show_keypoints:
        kpts0, kpts1 = np.round(kpts0).astype(int), np.round(kpts1).astype(int)
        white = (255, 255, 255)
        black = (0, 0, 0)
        for x, y in kpts0:
            cv2.circle(out, (x, y), 2, black, -1, lineType=cv2.LINE_AA)
            cv2.circle(out, (x, y), 1, white, -1, lineType=cv2.LINE_AA)
        for x, y in kpts1:
            cv2.circle(out, (x + margin + W0, y), 2, black, -1,
                       lineType=cv2.LINE_AA)
            cv2.circle(out, (x + margin + W0, y), 1, white, -1,
                       lineType=cv2.LINE_AA)

    mkpts0, mkpts1 = np.round(mkpts0).astype(int), np.round(mkpts1).astype(int)
    color = (np.array(color[:, :3])*255).astype(int)[:, ::-1]
    for (x0, y0), (x1, y1), c in zip(mkpts0, mkpts1, color):
        c = c.tolist()
        cv2.line(out, (x0, y0), (x1 + margin + W0, y1),
                 color=c, thickness=1, lineType=cv2.LINE_AA)
        # display line end-points as circles
        cv2.circle(out, (x0, y0), 2, c, -1, lineType=cv2.LINE_AA)
        cv2.circle(out, (x1 + margin + W0, y1), 2, c, -1,
                   lineType=cv2.LINE_AA)

    # Scale factor for consistent visualization across scales.
    sc = min(H / 640., 2.0)

    # Big text.
    Ht = int(30 * sc)  # text height
    txt_color_fg = (255, 255, 255)
    txt_color_bg = (0, 0, 0)
    for i, t in enumerate(text):
        cv2.putText(out, t, (int(8*sc), Ht*(i+1)), cv2.FONT_HERSHEY_DUPLEX,
                    1.0*sc, txt_color_bg, 2, cv2.LINE_AA)
        cv2.putText(out, t, (int(8*sc), Ht*(i+1)), cv2.FONT_HERSHEY_DUPLEX,
                    1.0*sc, txt_color_fg, 1, cv2.LINE_AA)

    # Small text.
    Ht = int(18 * sc)  # text height
    for i, t in enumerate(reversed(small_text)):
        cv2.putText(out, t, (int(8*sc), int(H-Ht*(i+.6))), cv2.FONT_HERSHEY_DUPLEX,
                    0.5*sc, txt_color_bg, 2, cv2.LINE_AA)
        cv2.putText(out, t, (int(8*sc), int(H-Ht*(i+.6))), cv2.FONT_HERSHEY_DUPLEX,
                    0.5*sc, txt_color_fg, 1, cv2.LINE_AA)

    if path is not None:
        cv2.imwrite(str(path), out)

    if opencv_display:
        cv2.imshow(opencv_title, out)
        cv2.waitKey(1)

    return out


def error_colormap(x):
    return np.clip(
        np.stack([2-x*2, x*2, np.zeros_like(x), np.ones_like(x)], -1), 0, 1)


# %%
import matplotlib.cm as cm
color = cm.jet(mconf)
text = ["",]
small_text=["","","",]
viz_path="a.png"
make_matching_plot(
    cv2.resize(imgA, (640,480)), 
    cv2.resize(imgB, (640,480)), kpts0, kpts1, mkpts0, mkpts1, color,
    text, viz_path, True,
    False, False, 'Matches', small_text)


# %%

# 対応店から
M, mask = cv2.findHomography(mkpts0, mkpts1, cv2.RANSAC, 5.0)

warpedA=cv2.warpPerspective(
    cv2.resize(imgA, (640,480)),
    M, dsize=(2048,2048))
warpedB=cv2.warpPerspective(
    cv2.resize(imgB, (640,480)), 
    #imgA,
    np.eye(3), dsize=(2048,2048))
plt.imshow(((warpedA/255.+warpedB/255.)/2.)[:700,:700])
plt.show()
