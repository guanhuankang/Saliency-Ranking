import numpy as np
import scipy.stats as stats
import copy

class Metrics:
    def __init__(self, metrics_of_interest = ["mae", "acc", "fbeta", "iou", "sa_sor", "sor", "ap", "ar", "top1"]):
        self.registerMetrics(metrics_of_interest)
    
    def registerMetrics(self, metrics_of_interest):
        self.metrics_of_interest = [m for m in metrics_of_interest if m in dir(self)]
        print("Register metrics: {}\nNot register metrics: {}".format(
            self.metrics_of_interest, 
            [x for x in metrics_of_interest if x not in self.metrics_of_interest]
        ), flush=True)
        
    def from_config(self, cfg):
        self.registerMetrics(cfg.TEST.METRICS_OF_INTEREST)
    
    def mergeMap(self, lst):
        merge = copy.deepcopy(lst[0])
        for m in lst[1::]:
            merge = np.maximum(merge, m)
        return merge

    def toNumpy(self, lst):
        return [x if isinstance(x, type(np.zeros((2,2)))) else np.array(x.detach().cpu()) for x in lst]

    def requireMatcher(self):
        require_matcher_metrics = ["ap", "ar", "sa_sor"]
        return len([1.0 for m in require_matcher_metrics if m in self.metrics_of_interest]) > 0

    def process(self, preds, gts, thres=.5):
        if len(gts) <= 0:
            print("warning GT has empty instances", flush=True)
            return {}
        if len(preds) <= 0:
            preds = [np.zeros_like(gts[0])]

        preds = self.toNumpy(preds if isinstance(preds, list) else [preds])
        gts = self.toNumpy(gts if isinstance(gts, list) else [gts])
        merge_pred = self.mergeMap(preds)
        merge_gt = self.mergeMap(gts)
        triples = self.matcher(preds, gts, thres=thres) if self.requireMatcher() else None
        results = {}
        for m in self.metrics_of_interest:
            results[m] = self.__getattribute__(m)(
                pred=merge_pred, 
                gt=merge_gt, 
                preds=preds, 
                gts=gts, 
                triples=triples, 
                thres=thres, 
                beta2=0.3
            )
        return results
    
    @classmethod
    def aggregate(self, results, reduction="mean"):
        ''' results: list of dict '''
        n = len(results)
        not_empty = 0
        reduction_keys = set()
        report = {}
        for i in range(n):
            for k in results[i]:
                s = results[i][k]
                if not np.isnan(s):
                    if k not in report: report[k] = 0.0 * s
                    report[k] += s
                    reduction_keys.add(k)
                else:
                    invalid_k = "invalid_"+k
                    if invalid_k not in report: report[invalid_k] = 0
                    report[invalid_k] += 1
            not_empty += 1 if len(results[i].keys()) > 0 else 0
        assert reduction in ["sum", "mean"], "reduction_{} only support sum, mean".format(reduction)
        if reduction=="mean":
            for k in reduction_keys:
                report[k] /= float(n)
        else:
            pass ## default SUM
        report.update({"not_empty": not_empty, "total": n})
        return report

    #####################-Numpy-Array-#########################
    ##############   Metrics Implementation       #############
    ##############   Coded By Huankang GUAN       #############
    ###########################################################
    '''
    All inputs (pred and gt) are maps with values between 0.0 and 1.0
    The default threshold is 0.5 if not specific
    '''
    def check(self, pred, gt):
        assert pred.shape==gt.shape, "shape of pred-{} and gt-{} are not matched".format(pred.shape, gt.shape)
        assert pred.max()<=1.0 and pred.min()>=0.0, "max-{}/min-{} value of pred is not btw 0 and 1".format(pred.max(), pred.min())
        assert gt.max()<=1.0 and gt.min()>=0.0, "max-{}/min-{} value of gt is not btw 0 and 1".format(gt.max(), gt.min())

    def matcher(self, preds, gts, thres=.5, **argw):
        ''' return: list of dict likes:
            {
                "iou": 0.95,
                "pred": np.array/None,
                "gt": np.array/None,
                "pred_rank": None:0 Rank:1-N (saliency: low -> high)
                "gt_rank": None:0 Rank:1-N (saliency: low -> high)
            }
            where None means matching none
        '''
        triples = [(self.iou(p,g,thres), i, j) for i,p in enumerate(preds) for j,g in enumerate(gts)]
        triples.sort(key=lambda t:t[0], reverse=True)
        ret = []
        used_p = []
        used_g = []
        n_pred = len(preds)
        n_gt = len(gts)
        for t in triples:
            iou, i, j = t
            if (i not in used_p) and (j not in used_g):
                ret.append({"iou":iou,"pred":preds[i],"gt":gts[j],"pred_rank":n_pred-i,"gt_rank":n_gt-j})
                used_p.append(i)
                used_g.append(j)
        for i in range(len(preds)):
            if i not in used_p:
                ret.append({"iou":0.0,"pred":preds[i],"gt":None,"pred_rank":n_pred-i,"gt_rank":0})
                used_p.append(i)
        for j in range(len(gts)):
            if j not in used_g:
                ret.append({"iou":0.0,"pred":None,"gt":gts[j],"pred_rank":0,"gt_rank":n_gt-j})
                used_g.append(j)
        assert len(used_g)==len(gts) and len(preds)==len(used_p)
        return ret

    def iou(self, pred, gt, thres=.5, **argw):
        self.check(pred, gt)
        inter = np.logical_and(pred>thres, gt>thres).sum()
        union = np.logical_or(pred>thres, gt>thres).sum()
        return inter / (union + 1e-6)

    def mae(self, pred, gt, **argw):
        self.check(pred, gt)
        return np.mean(np.abs(pred.astype(float) - gt.astype(float)))

    def acc(self, pred, gt, thres=.5, **argw):
        self.check(pred, gt)
        p = pred > thres
        g = gt > thres
        return 1.0 - float(np.logical_xor(p, g).sum())/float(np.prod(p.shape))

    def fbeta(self, pred, gt, thres=.5, beta2=0.3, **argw):
        self.check(pred, gt)
        p = (pred > thres) * 1.0
        g = (gt > thres) * 1.0
        tp = (p * g).sum()
        fp = (p * (1.0-g)).sum()
        fn = ((1.0-p) * g).sum()
        pre = tp / (tp + fp + 1e-6)
        rec = tp / (tp + fn + 1e-6)
        return ( (1.+beta2) * pre * rec) / ( beta2 * pre + rec + 1e-6 )

    def ap(self, triples, thres=0.5, **argw):
        n_hit = sum([1.0 for t in triples if t["iou"]>thres])
        n_pred = sum([1.0 for t in triples if not isinstance(t["pred"], type(None))])
        n_gt = sum([1.0 for t in triples if not isinstance(t["gt"], type(None))])
        if n_gt>0:
            return n_hit / (n_pred + 1e-6)
        else:
            return 0.0 if n_pred>0.0 else 1.0
    
    def ar(self, triples, thres=.5, **argw):
        n_hit = sum([1.0 for t in triples if t["iou"]>thres])
        n_gt = sum([1.0 for t in triples if not isinstance(t["gt"], type(None))])
        if n_gt>0:
            return n_hit / n_gt
        else:
            return 1.0

    def sor(self, preds, gts, thres=.5, **argw):
        gt_ranks = []
        pred_ranks = []
        n_gt = len(gts)
        n_pred = len(preds)
        for i in range(n_gt):
            for j in range(n_pred):
                if self.iou(preds[j], gts[i]) > thres:
                    gt_ranks.append(n_gt - i)
                    pred_ranks.append(n_pred - j)
                    break
        if len(gt_ranks) > 1:
            try:
                spr = stats.spearmanr(pred_ranks, gt_ranks).statistic
            except:
                spr = stats.spearmanr(pred_ranks, gt_ranks).correlation
            return (spr + 1.0)/2.0
        elif len(gt_ranks) == 1:
            return 1.0
        else:
            return np.nan
    
    def sa_sor(self, triples, thres=.5, **argw):
        pred_ranks = []
        gt_ranks = []
        for t in triples:
            if isinstance(t["gt"], type(None)): continue
            pred_ranks.append(t["pred_rank"] if t["iou"]>thres else 0)
            gt_ranks.append(t["gt_rank"])
        return np.corrcoef(pred_ranks, gt_ranks)[0, 1]

    def top1(self, preds, gts, thres=.5, **argw):
        if len(gts) > 0 and len(preds) > 0:
            return self.iou(pred=preds[0], gt=gts[0], thres=thres)
        return 0.0

    def top2(self, preds, gts, thres=.5, **argw):
        k = 2
        if len(gts) >= k and len(preds) >= k:
            return self.iou(pred=preds[k - 1], gt=gts[k - 1], thres=thres)
        elif len(gts) < k and len(preds) < k:
            return 1.0
        return 0.0

    def top3(self, preds, gts, thres=.5, **argw):
        k = 3
        if len(gts) >= k and len(preds) >= k:
            return self.iou(pred=preds[k - 1], gt=gts[k - 1], thres=thres)
        elif len(gts) < k and len(preds) < k:
            return 1.0
        return 0.0

    def top4(self, preds, gts, thres=.5, **argw):
        k = 4
        if len(gts) >= k and len(preds) >= k:
            return self.iou(pred=preds[k - 1], gt=gts[k - 1], thres=thres)
        elif len(gts) < k and len(preds) < k:
            return 1.0
        return 0.0

    def top5(self, preds, gts, thres=.5, **argw):
        k = 5
        if len(gts) >= k and len(preds) >= k:
            return self.iou(pred=preds[k - 1], gt=gts[k - 1], thres=thres)
        elif len(gts) < k and len(preds) < k:
            return 1.0
        return 0.0


if __name__=="__main__":
    import os, tqdm
    from PIL import Image

    def decompose(m, ignore=[0]):
        vals = np.unique(m)[::-1]
        rets = []
        for v in vals:
            if v in ignore: continue
            rets.append(np.where(m==v, 1.0, 0.0).astype(float))
        return rets

    metrics = Metrics(metrics_of_interest = ["mae", "acc", "fbeta", "iou", "sa_sor", "sor", "ap", "ar"])
    input_path = r"D:\SaliencyRanking\retrain_compared_results\IRSR\IRSR\prediction"
    output_path = r"D:\SaliencyRanking\dataset\irsr\Images\test\gt"
    names = [x for x in os.listdir(output_path) if x.endswith(".png")]
    results = []
    for name in tqdm.tqdm(names):
        if os.path.exists(os.path.join(input_path, name)):
            preds = decompose(np.array(Image.open(os.path.join(input_path, name)).convert("L"), dtype=np.uint8))
        if len(preds)<=0:
            preds = [np.zeros((480, 640), dtype=float)]
        gts = decompose(np.array(Image.open(os.path.join(output_path, name)).convert("L"), dtype=np.uint8))
        results.append(metrics.process(preds, gts, thres=.5))
    report = metrics.aggregate(results)
    print(report)