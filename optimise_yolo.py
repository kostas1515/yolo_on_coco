from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
from bayes_opt.util import load_logs
import bayesian_pipeline as bayesian_pipeline
import numpy as np


pbounds = {'lr':(1e-7,0.0001),'w': (1e-7, 0.001),'m': (0.8, 0.999),'g':(0.0, 2.0),'a':(0.01, 0.99),
           'lcoor':(1e-7,1.0),'lno':(1e-7,1.0),'iou_thresh':(0.2,0.75),'iou_type':(0.0,4.0),'inf_c':(1e-7,0.1),'inf_t':(0.25,0.75)}



for i in range(25):
    optimizer = BayesianOptimization(
        f=bayesian_pipeline.bayesian_opt,
        pbounds=pbounds,
        verbose=2, # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
        random_state=i,
    )

    optimizer.maximize(
        init_points=10,
        n_iter=1,
    )

    print(optimizer.max)

    params=optimizer.max['params']

    bayesian_pipeline.bayesian_opt(params['lr'],params['w'],params['m'],params['g'],params['a'],
                                   params['lcoor'],params['lno'],params['iou_thresh'],params['iou_type'],
                                   params['inf_c'],params['inf_t'],bayes_opt=False)

