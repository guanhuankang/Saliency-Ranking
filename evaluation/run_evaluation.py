from evaluation import Evaluation

eval = Evaluation()
irsr_gt = r"D:\SaliencyRanking\dataset\irsr\Images\test\gt"
assr_gt = r"D:\SaliencyRanking\dataset\ASSR\ASSR\gt\test"

## 2022-TPAMI IRSR
pred_irsr = r"D:\SaliencyRanking\retrain_compared_results\IRSR\IRSR\prediction"
pred_assr = r"D:\SaliencyRanking\retrain_compared_results\IRSR\ASSR\prediction"
eval(test_name="IRSRonIRSR", pred_path=pred_irsr, gt_path=irsr_gt)
eval(test_name="IRSRonASSR", pred_path=pred_assr, gt_path=assr_gt)

## 2021-ICCV PPA
pred_irsr = r"D:\SaliencyRanking\retrain_compared_results\PPA\IRSR\saliency_map"
pred_assr = r"D:\SaliencyRanking\retrain_compared_results\PPA\ASSR\saliency_map"
eval(test_name="PPAonIRSR", pred_path=pred_irsr, gt_path=irsr_gt)
eval(test_name="PPAonASSR", pred_path=pred_assr, gt_path=assr_gt)

exit(0)

## 2020-CVPR ASSR
pred_irsr = ""
pred_assr = ""
eval(test_name="ASSRonIRSR", pred_path=pred_irsr, gt_path=irsr_gt)
eval(test_name="ASSRonASSR", pred_path=pred_assr, gt_path=assr_gt)

## 2018-CVPR RSDNet
pred_irsr = ""
pred_assr = ""
eval(test_name="RSDNetonIRSR", pred_path=pred_irsr, gt_path=irsr_gt)
eval(test_name="RSDNetonASSR", pred_path=pred_assr, gt_path=assr_gt)

