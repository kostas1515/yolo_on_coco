from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
import yolo_function as yolo_function



logger = JSONLogger(path="../bayesian_opt_logger/yolo_v2.json")


pbounds = {'weight_decay': (0, 0.01),'gamma':(0.0, 2.0), 'alpha':(0.0,1.0),
           'lcoord':(1.0,10.0),'lno_obj':(0.01,1.0),'iou_ignore_thresh':(0.4,0.7),'iou_type':(0.0,4.0),'tf_idf':(0.0,1.0)}


optimizer = BayesianOptimization(
    f=yolo_function.train_yolo,
    pbounds=pbounds,
    verbose=2, # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
    random_state=0,
)
optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)
optimizer.maximize(
    init_points=30,
    n_iter=10,
)
