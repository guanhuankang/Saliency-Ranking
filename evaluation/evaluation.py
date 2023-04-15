import os
import datetime
import numpy as np
from PIL import Image
from scipy import stats
import datetime
import tqdm, json

def loadJson(file):
    if not os.path.exists(file):
        return {}
    with open(file, "r") as f:
        data = json.load(f)
    return data

def dumpJson(data, file):
    with open(file, "w") as f:
        json.dump(data, f)

class Evaluation:
    def __init__(self):
        super().__init__()

    def IOU(self, pred, gt):
        assert pred.shape==gt.shape
        pred = (pred * 1).astype(np.int32)
        gt = (gt * 1).astype(np.int32)
        inter = np.logical_and(pred>0, gt>0).sum()
        union = np.logical_or(pred>0, gt>0).sum()
        return inter / (union + 1e-6)

    def saSOR(self, pred, gt, thres = 0.5):
        BG = 0
        DTYPE = np.uint8
        assert pred.shape==gt.shape
        assert pred.dtype==gt.dtype and pred.dtype==DTYPE

        gt_uni = np.unique(np.append(np.unique(gt), BG))
        gt_map = dict( (x,r) for r,x in enumerate(gt_uni) )
        pred_uni = np.unique(np.append(np.unique(pred), BG))
        pred_map = dict( (x,r) for r,x in enumerate(pred_uni) )

        matrixs = []
        for gval in gt_uni:
            if gval==BG: continue
            for pval in pred_uni:
                if pval==BG: continue
                iou = self.IOU(pred==pval, gt==gval)
                if iou >= thres:
                    matrixs.append( (iou, gt_map[gval], pred_map[pval]) )
        matrixs.sort(reverse=True)

        gt_rank = []
        pred_rank = []
        for item in matrixs:
            iou, g_r, p_r = item
            if (g_r not in gt_rank) and (p_r not in pred_rank):
                gt_rank.append(g_r)
                pred_rank.append(p_r)

        for r in range(1, len(gt_uni)):
            if r not in gt_rank:
                gt_rank.append(r)
                pred_rank.append(0)

        return np.corrcoef(pred_rank, gt_rank)[0, 1]

    def mae(self, pred, gt):
        DTYPE = np.uint8
        assert pred.shape==gt.shape
        assert pred.dtype==gt.dtype and pred.dtype==DTYPE
        p = pred.astype(float) / 255.
        g = gt.astype(float) / 255.
        return np.mean(np.abs(p - g))

    def acc(self, pred, gt):
        DTYPE = np.uint8
        assert pred.shape==gt.shape
        assert pred.dtype==gt.dtype and pred.dtype==DTYPE
        p = (pred > 0.0) * 1.0
        g = (gt > 0.0) * 1.0
        return 1.0 - np.mean(np.abs(p - g))

    def __call__(self, name, pred_path, gt_path):
        lst = [name for name in os.listdir(gt_path) if name.endswith(".png")]
        print("#test_set={}".format(len(lst)), flush=True)

        saSor_scores = []
        mae_scores = []
        acc_scores = []
        valid = 0
        for name in tqdm.tqdm(lst):
            gt = np.array(Image.open(os.path.join(gt_path, name)).convert("L"))
            if os.path.exists(os.path.join(pred_path, name)):
                pred = np.array(Image.open(os.path.join(pred_path, name)).convert("L"))
            else:
                pred = np.zeros_like(gt)

            mae_scores.append(self.mae(pred, gt))
            acc_scores.append(self.acc(pred, gt))
            coff = self.saSOR(pred, gt)
            if np.isnan(coff):
                saSor_scores.append(0)
            else:
                saSor_scores.append(coff)
                valid += 1

        ## save results
        file_name = "evaluation.json"
        time_indice = str(datetime.datetime.now()).replace(" ", "_")
        history = loadJson(file_name)
        history[time_indice] = {
            "name": name,
            "len": len(lst),
            "valid": valid,
            "accuracy": np.mean(acc_scores),
            "mae": np.mean(mae_scores),
            "SA-SOR(zero)": np.mean(saSor_scores),
            "SA-SOR(valid)": np.sum(saSor_scores) / valid
        }
        print(history[time_indice], flush=True)
        dumpJson(history, file_name)

if __name__=="__main__":
    eval = Evaluation()
    eval(
        name="IRSRonIRSR",
        pred_path=r"D:\SaliencyRanking\retrain_compared_results\IRSR\IRSR\prediction",
        gt_path=r"D:\SaliencyRanking\dataset\irsr\Images\test\gt"
    )
    eval(
        name="PPAonASSR",
        pred_path=r"D:\SaliencyRanking\retrain_compared_results\PPA\ASSR\saliency_map",
        gt_path=r"D:\SaliencyRanking\dataset\ASSR\ASSR\gt\test"
    )