import pickle
from collections import Counter

import numpy as np


def str2ind(categoryname, classlist):
    return [i for i in range(len(classlist)) if categoryname == classlist[i]][0]


def encode_mask_to_rle(mask):
    """
    mask: numpy array binary mask
    1 - mask
    0 - background
    Returns encoded run length
    """
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return runs


def filter_segments(segment_predict, videonames, ambilist, factor):
    ind = np.zeros(np.shape(segment_predict)[0])
    for i in range(np.shape(segment_predict)[0]):
        vn = videonames[int(segment_predict[i, 0])]
        for a in ambilist:
            if a[0] == vn:
                gt = range(
                    int(round(float(a[2]) * factor)), int(round(float(a[3]) * factor))
                )
                pd = range(int(segment_predict[i][1]), int(segment_predict[i][2]))
                IoU = float(len(set(gt).intersection(set(pd)))) / float(
                    len(set(gt).union(set(pd)))
                )
                if IoU > 0:
                    ind[i] = 1
    s = [
        segment_predict[i, :]
        for i in range(np.shape(segment_predict)[0])
        if ind[i] == 0
    ]
    return np.array(s)


def getActLoc(
    vid_preds, frm_preds, vid_lens, act_thresh_cas, annotation_path, args, multi=False
):

    try:
        with open(annotation_path) as f:
            data = pickle.load(f)
    except:
        # for pickle file from python2
        with open(annotation_path, "rb") as f:
            data = pickle.load(f, encoding="latin1")

    if multi:
        gtsegments = []
        gtlabels = []
        for idx in range(len(data["L"])):
            gt = data["L"][idx]
            gt_ = set(gt)
            gt_.discard(args.model_args["num_class"])
            gts = []
            gtl = []
            for c in list(gt_):
                gt_encoded = encode_mask_to_rle(gt == c)
                gts.extend(
                    [
                        [x - 1, x + y - 2]
                        for x, y in zip(gt_encoded[::2], gt_encoded[1::2])
                    ]
                )
                gtl.extend([c for item in gt_encoded[::2]])
            gtsegments.append(gts)
            gtlabels.append(gtl)
    else:
        gtsegments = []
        gtlabels = []
        for idx in range(len(data["L"])):
            gt = data["L"][idx]
            gt_encoded = encode_mask_to_rle(gt)
            gtsegments.append(
                [[x - 1, x + y - 2] for x, y in zip(gt_encoded[::2], gt_encoded[1::2])]
            )
            gtlabels.append([data["Y"][idx] for item in gt_encoded[::2]])

    videoname = np.array(data["sid"])

    # keep ground truth and predictions for instances with temporal annotations
    gtl, vn, vp, fp, vl = [], [], [], [], []
    for i, s in enumerate(gtsegments):
        if len(s):
            gtl.append(gtlabels[i])
            vn.append(videoname[i])
            vp.append(vid_preds[i])
            fp.append(frm_preds[i])
            vl.append(vid_lens[i])
    gtlabels = gtl
    videoname = vn

    # which categories have temporal labels ?
    templabelidx = sorted(list(set([l for gtl in gtlabels for l in gtl])))

    dataset_segment_predict = []
    class_threshold = args.class_threshold
    for c in range(frm_preds[0].shape[1]):
        c_temp = []
        # Get list of all predictions for class c
        for i in range(len(fp)):
            vid_cls_score = vp[i][c]
            vid_cas = fp[i][:, c]
            vid_cls_proposal = []
            if vid_cls_score < class_threshold:
                continue
            for t in range(len(act_thresh_cas)):
                thres = act_thresh_cas[t]
                vid_pred = np.concatenate(
                    [np.zeros(1), (vid_cas > thres).astype("float32"), np.zeros(1)],
                    axis=0,
                )
                vid_pred_diff = [
                    vid_pred[idt] - vid_pred[idt - 1] for idt in range(1, len(vid_pred))
                ]
                s = [idk for idk, item in enumerate(vid_pred_diff) if item == 1]
                e = [idk for idk, item in enumerate(vid_pred_diff) if item == -1]
                for j in range(len(s)):
                    len_proposal = e[j] - s[j]
                    if len_proposal >= 3:
                        inner_score = np.mean(vid_cas[s[j] : e[j] + 1])
                        outer_s = max(0, int(s[j] - 0.25 * len_proposal))
                        outer_e = min(
                            int(vid_cas.shape[0] - 1),
                            int(e[j] + 0.25 * len_proposal + 1),
                        )
                        outer_temp_list = list(range(outer_s, int(s[j]))) + list(
                            range(int(e[j] + 1), outer_e)
                        )
                        if len(outer_temp_list) == 0:
                            outer_score = 0
                        else:
                            outer_score = np.mean(vid_cas[outer_temp_list])
                        c_score = inner_score - 0.6 * outer_score
                        vid_cls_proposal.append([i, s[j], e[j] + 1, c_score])
            pick_idx = NonMaximumSuppression(np.array(vid_cls_proposal), 0.2)
            nms_vid_cls_proposal = [vid_cls_proposal[k] for k in pick_idx]
            c_temp += nms_vid_cls_proposal
        if len(c_temp) > 0:
            c_temp = np.array(c_temp)
        dataset_segment_predict.append(c_temp)
    """
    for i, pred in enumerate(dataset_segment_predict):
        print (f"#{i} class {c} has {len(pred)} predictions")
    """
    return dataset_segment_predict


def IntergrateSegs(rgb_segs, flow_segs, th, args):
    NUM_CLASS = args.class_num
    NUM_VID = 212
    segs = []
    for i in range(NUM_CLASS):
        class_seg = []
        rgb_seg = rgb_segs[i]
        flow_seg = flow_segs[i]
        rgb_seg_ind = np.array(rgb_seg)[:, 0]
        flow_seg_ind = np.array(flow_seg)[:, 0]
        for j in range(NUM_VID):
            rgb_find = np.where(rgb_seg_ind == j)
            flow_find = np.where(flow_seg_ind == j)
            if len(rgb_find[0]) == 0 and len(flow_find[0]) == 0:
                continue
            elif len(rgb_find[0]) != 0 and len(flow_find[0]) != 0:
                rgb_vid_seg = rgb_seg[rgb_find[0]]
                flow_vid_seg = flow_seg[flow_find[0]]
                fuse_seg = np.concatenate([rgb_vid_seg, flow_vid_seg], axis=0)
                pick_idx = NonMaximumSuppression(fuse_seg, th)
                fuse_segs = fuse_seg[pick_idx]
                class_seg.append(fuse_segs)
            elif len(rgb_find[0]) != 0 and len(flow_find[0]) == 0:
                vid_seg = rgb_seg[rgb_find[0]]
                class_seg.append(vid_seg)
            elif len(rgb_find[0]) == 0 and len(flow_find[0]) != 0:
                vid_seg = flow_seg[flow_find[0]]
                class_seg.append(vid_seg)
        class_seg = np.concatenate(class_seg, axis=0)
        segs.append(class_seg)
    return segs


def NonMaximumSuppression(segs, overlapThresh):
    # if there are no boxes, return an empty list
    if len(segs) == 0:
        return []
    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if segs.dtype.kind == "i":
        segs = segs.astype("float")

    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the segments
    s = segs[:, 1]
    e = segs[:, 2]
    scores = segs[:, 3]
    # compute the area of the bounding boxes and sort the bounding
    # boxes by the score of the bounding box
    area = e - s + 1
    idxs = np.argsort(scores)

    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest coordinates for the start of
        # the segments and the smallest coordinates
        # for the end of the segments
        maxs = np.maximum(s[i], s[idxs[:last]])
        mine = np.minimum(e[i], e[idxs[:last]])

        # compute the length of the overlapping area
        l = np.maximum(0, mine - maxs + 1)
        # compute the ratio of overlap
        overlap = l / area[idxs[:last]]

        # delete segments beyond the threshold
        idxs = np.delete(
            idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0]))
        )
    return pick


def getLocMAP(seg_preds, th, annotation_path, args, multi=False, factor=1.0):
    try:
        with open(annotation_path) as f:
            data = pickle.load(f)
    except:
        # for pickle file from python2
        with open(annotation_path, "rb") as f:
            data = pickle.load(f, encoding="latin1")

    if multi:
        gtsegments = []
        gtlabels = []
        for idx in range(len(data["L"])):
            gt = data["L"][idx]
            gt_ = set(gt)
            gt_.discard(args.model_args["num_class"])
            gts = []
            gtl = []
            for c in list(gt_):
                gt_encoded = encode_mask_to_rle(gt == c)
                gts.extend(
                    [
                        [x - 1, x + y - 2]
                        for x, y in zip(gt_encoded[::2], gt_encoded[1::2])
                    ]
                )
                gtl.extend([c for item in gt_encoded[::2]])
            gtsegments.append(gts)
            gtlabels.append(gtl)
    else:
        gtsegments = []
        gtlabels = []
        for idx in range(len(data["L"])):
            gt = data["L"][idx]
            gt_encoded = encode_mask_to_rle(gt)
            gtsegments.append(
                [[x - 1, x + y - 2] for x, y in zip(gt_encoded[::2], gt_encoded[1::2])]
            )
            gtlabels.append([data["Y"][idx] for item in gt_encoded[::2]])

    videoname = np.array(data["sid"])
    """
    cnt = Counter(data['Y'])
    d = cnt.most_common()
    print (d)
    """
    # which categories have temporal labels ?
    templabelidx = sorted(list(set([l for gtl in gtlabels for l in gtl])))

    ap = []
    for c in templabelidx:
        segment_predict = seg_preds[c]
        # Sort the list of predictions for class c based on score
        if len(segment_predict) == 0:
            ap.append(0.0)
            continue
        segment_predict = segment_predict[np.argsort(-segment_predict[:, 3])]

        # Create gt list
        segment_gt = [
            [i, gtsegments[i][j][0], gtsegments[i][j][1]]
            for i in range(len(gtsegments))
            for j in range(len(gtsegments[i]))
            if gtlabels[i][j] == c
        ]
        gtpos = len(segment_gt)

        # Compare predictions and gt
        tp, fp = [], []
        for i in range(len(segment_predict)):
            matched = False
            best_iou = 0
            for j in range(len(segment_gt)):
                if segment_predict[i][0] == segment_gt[j][0]:
                    gt = range(
                        int(round(segment_gt[j][1] * factor)),
                        int(round(segment_gt[j][2] * factor)),
                    )
                    p = range(int(segment_predict[i][1]), int(segment_predict[i][2]))
                    IoU = float(len(set(gt).intersection(set(p)))) / float(
                        len(set(gt).union(set(p)))
                    )
                    if IoU >= th:
                        matched = True
                        if IoU > best_iou:
                            best_iou = IoU
                            best_j = j
            if matched:
                del segment_gt[best_j]
            tp.append(float(matched))
            fp.append(1.0 - float(matched))
        tp_c = np.cumsum(tp)
        fp_c = np.cumsum(fp)
        # print (c, tp, fp)
        if sum(tp) == 0:
            prc = 0.0
        else:
            cur_prec = tp_c / (fp_c + tp_c)
            cur_rec = tp_c / gtpos
            prc = _ap_from_pr(cur_prec, cur_rec)
        ap.append(prc)

    print(f" ".join([f"{item*100:.2f}" for item in ap]))
    if ap:
        return 100 * np.mean(ap)
    else:
        return 0


# Inspired by Pascal VOC evaluation tool.
def _ap_from_pr(prec, rec):
    mprec = np.hstack([[0], prec, [0]])
    mrec = np.hstack([[0], rec, [1]])

    for i in range(len(mprec) - 1)[::-1]:
        mprec[i] = max(mprec[i], mprec[i + 1])

    idx = np.where(mrec[1::] != mrec[0:-1])[0] + 1
    ap = np.sum((mrec[idx] - mrec[idx - 1]) * mprec[idx])

    return ap


def compute_iou(dur1, dur2):
    # find the each edge of intersect rectangle
    left_line = max(dur1[0], dur2[0])
    right_line = min(dur1[1], dur2[1])

    # judge if there is an intersect
    if left_line >= right_line:
        return 0
    else:
        intersect = right_line - left_line
        union = max(dur1[1], dur2[1]) - min(dur1[0], dur2[0])
        return intersect / union


def getSingleStreamDetectionMAP(
    vid_preds, frm_preds, vid_lens, annotation_path, args, multi=False, factor=1.0
):
    iou_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    dmap_list = []
    seg = getActLoc(
        vid_preds,
        frm_preds,
        vid_lens,
        np.arange(args.start_threshold, args.end_threshold, args.threshold_interval),
        annotation_path,
        args,
        multi=multi,
    )
    # print (len(seg))
    for iou in iou_list:
        print("Testing for IoU %f" % iou)
        dmap_list.append(
            getLocMAP(seg, iou, annotation_path, args, multi=multi, factor=factor)
        )
    return dmap_list, iou_list


def getTwoStreamDetectionMAP(
    rgb_vid_preds,
    flow_vid_preds,
    rgb_frm_preds,
    flow_frm_preds,
    vid_lens,
    annotation_path,
    args,
):
    iou_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    dmap_list = []
    rgb_seg = getActLoc(
        rgb_vid_preds,
        rgb_frm_preds * 0.1,
        vid_lens,
        np.arange(args.start_threshold, args.end_threshold, args.threshold_interval)
        * 0.1,
        annotation_path,
        args,
    )
    flow_seg = getActLoc(
        flow_vid_preds,
        flow_frm_preds,
        vid_lens,
        np.arange(args.start_threshold, args.end_threshold, args.threshold_interval),
        annotation_path,
        args,
    )
    seg = IntergrateSegs(rgb_seg, flow_seg, 0.9, args)
    for iou in iou_list:
        print("Testing for IoU %f" % iou)
        dmap_list.append(getLocMAP(seg, iou, annotation_path, args))

    return dmap_list, iou_list
