from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
from bayes_opt.util import load_logs
import bayesian_pipeline as bayesian_pipeline
import numpy as np


pbounds = {'w': (0, 0.001),'m': (0.8, 1.0),'g':(0.0, 2.0),'a':(0.01, 0.99),
           'lcoor':(0.0,10.0),'lno':(0.01,1.0),'iou_thresh':(0.1,0.9),'iou_type':(0.0,4.0),'inf_c':(0.0,1.0),'inf_t':(0.0,1.0)}



for i in range(25):
    optimizer = BayesianOptimization(
        f=bayesian_pipeline.bayesian_opt,
        pbounds=pbounds,
        verbose=1, # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
        random_state=i,
    )

    optimizer.maximize(
        init_points=20,
        n_iter=2,
    )

    print(optimizer.max)

    params=optimizer.max['params']

    bayesian_pipeline.bayesian_opt(params['w'],params['m'],params['g'],params['a'],
                                   params['lcoor'],params['lno'],params['iou_thresh'],params['iou_type'],
                                   params['inf_c'],params['inf_t'],bayes_opt=False)

