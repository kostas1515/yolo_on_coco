from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
from bayes_opt.util import load_logs
import bayesian_pipeline as bayesian_pipeline
import numpy as np


pbounds = {'w': (1e-7, 0.01),'m': (0.6, 0.999),'g':(0.0, 2.0),'a':(0.01, 0.99),
           'lcoor':(1e-7,10.0),'lno':(1e-7,5.0),'iou_thresh':(0.1,0.95),'iou_type':(0.0,4.0)}




optimizer = BayesianOptimization(
    f=bayesian_pipeline.bayesian_opt,
    pbounds=pbounds,
    verbose=2, # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
    random_state=64,
)

optimizer.maximize(
    init_points=20,
    n_iter=10,
)

print(optimizer.max)

#     params=optimizer.max['params']

#     bayesian_pipeline.bayesian_opt(params['lr'],params['w'],params['m'],params['g'],params['a'],
#                                    params['lcoor'],params['lno'],params['iou_thresh'],params['iou_type'],
#                                    params['inf_c'],params['inf_t'],bayes_opt=False)

