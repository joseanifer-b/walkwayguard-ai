# Dataset folder

Extract the datasets into these folders (the empty `PUT_..._HERE` files are placeholders - delete them if you like):

    dataset/avenue/          <- CUHK Avenue         (Kaggle link: ______________________)
    dataset/shanghaitech/    <- ShanghaiTech Campus (Kaggle link: ______________________)

The loader tolerates the different Kaggle layouts. It needs:

* **Training sequences** (normal footage): videos (.avi/.mp4) or folders of frames under a path containing `train`
  (e.g. `training_videos/`, `training/frames/`).
* **Testing sequences**: the same under a path containing `test`.
* **Ground truth** (only for evaluation - not needed to train or analyse videos), matched to a test sequence by name:
  * Avenue: `testing_label_mask/<n>_label.mat` (variable `volLabel`) for test video `<n>.avi`
  * ShanghaiTech: `test_frame_mask/<seq>.npy` (per-frame 0/1), optionally `test_pixel_mask/<seq>.npy`
  * any dataset: `<seq>.npy` or `<seq>_label.npy` (1-D per-frame labels, or N x H x W masks)

The dashboard's **Dashboard -> Datasets** card shows how many train / test / labelled sequences were found.
