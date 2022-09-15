import math
import numpy as np
from IPython.core.debugger import set_trace

def reject_by_metric(predictions, labels, metric, rejection_rates, reverse_order=False):
    accs = []

    for rejection_rate in rejection_rates:
        inv_ranks_rejected = []
        
        sorting = np.argsort(metric)
        if reverse_order:
            sorting = np.flip(sorting)
            
        sorted_predicts = np.array(predictions)[sorting]
        sorted_answers = np.array(labels)[sorting]

        upper_bound = math.ceil((len(sorted_predicts) * rejection_rate).item())
        predicts_after_reject = sorted_predicts[0:upper_bound]
        answers_after_reject = sorted_answers[0:upper_bound]

        for predicts, a in zip(predicts_after_reject, answers_after_reject):
            inv_rs = []
            if len(predicts) > 0:
                for a_i in a:
                    if a_i not in predicts:
                        inv_r = 0
                    else:
                        inv_r = 1 / (list(predicts).index(a_i) + 1)
                    inv_rs.append(inv_r)
                inv_ranks_rejected.append(max(inv_rs))
            else:
                inv_ranks_rejected.append(0)

        top1 = np.array(inv_ranks_rejected)[np.array(inv_ranks_rejected) == 1]
        accs.append(len(top1) / len(inv_ranks_rejected))
    
    return accs