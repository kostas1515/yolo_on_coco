from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
import yolo_function as yolo_function
from bayes_opt.util import load_logs


pbounds = {'weight_decay': (0, 0.001),'momentum': (0.8, 1.0),'gamma':(0.0, 2.0),
           'lcoord':(1.0,10.0),'lno_obj':(0.01,1.0),'iou_ignore_thresh':(0.2,0.7),'iou_type':(0.0,4.0)}




optimizer = BayesianOptimization(
    f=yolo_function.train_yolo,
    pbounds=pbounds,
    verbose=2, # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
    random_state=88,
)

load_logs(optimizer, logs=["../bayesian_opt_logger/logs.json"])

logger = JSONLogger(path="../bayesian_opt_logger/logs.json")

optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)

optimizer.maximize(
    init_points=10,
    n_iter=10,
)

print(optimizer.max)
