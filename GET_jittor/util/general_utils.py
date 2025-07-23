import os
import inspect
import jittor as jt
from datetime import datetime
from loguru import logger

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):

        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):

        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def init_experiment(args, runner_name=None, exp_id=None):
    # Get filepath of calling script
    if runner_name is None:
        runner_name = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))).split(".")[-2:]

    root_dir = os.path.join(args.exp_root, *runner_name)

    if not os.path.exists(root_dir):
        os.makedirs(root_dir)

    # Either generate a unique experiment ID, or use one which is passed
    if exp_id is None:

        if args.exp_name is None:
            raise ValueError("Need to specify the experiment name")
        # Unique identifier for experiment
        now = '{}_({:02d}.{:02d}.{}_|_'.format(args.exp_name, datetime.now().day, datetime.now().month, datetime.now().year) + \
              datetime.now().strftime("%S.%f")[:-3] + ')'

        log_dir = os.path.join(root_dir, 'log', now)
        while os.path.exists(log_dir):
            now = '({:02d}.{:02d}.{}_|_'.format(datetime.now().day, datetime.now().month, datetime.now().year) + \
                  datetime.now().strftime("%S.%f")[:-3] + ')'

            log_dir = os.path.join(root_dir, 'log', now)

    else:

        log_dir = os.path.join(root_dir, 'log', f'{exp_id}')

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
        
    logger.add(os.path.join(log_dir, 'log.txt'))
    args.logger = logger
    args.log_dir = log_dir

    # Instantiate directory to save models to
    model_root_dir = os.path.join(args.log_dir, 'checkpoints')
    if not os.path.exists(model_root_dir):
        os.mkdir(model_root_dir)

    args.model_dir = model_root_dir
    args.tes_model_path = os.path.join(args.model_dir, 'tes_model.pkl')
    args.model_path = os.path.join(args.model_dir, 'model.pkl')

    print(f'Experiment saved to: {args.log_dir}')

    hparam_dict = {}

    for k, v in vars(args).items():
        if isinstance(v, (int, float, str, bool, jt.Var)):
            hparam_dict[k] = v

    print(runner_name)
    print(args)

    return args



    
    
def distill_crit(stu_feats, tea_feats, labels, args=None, device="cuda"):
    # 确保在 Jittor 下张量都是 float32
    stu_feats = jt.array(stu_feats).float()
    tea_feats = jt.array(tea_feats).float()
    labels = jt.array(labels).int()

    similarity_matrix = jt.matmul(stu_feats, tea_feats.transpose(1, 0))

    num_labels = similarity_matrix.shape[0]
    dim_labels = similarity_matrix.shape[1]

    labels_one_hot = jt.zeros((num_labels, dim_labels))
    labels_one_hot = labels_one_hot.scatter(1, labels.unsqueeze(1), jt.array(1))

    positives = similarity_matrix[labels_one_hot.bool()].reshape(num_labels, -1)
    negatives = similarity_matrix[(1-labels_one_hot).bool()].reshape(num_labels, -1)

    logits = jt.concat([positives, negatives], dim=1)
    labels = jt.zeros((logits.shape[0],), dtype=jt.int32)

    # 可选温度缩放
    temperature = getattr(args, 'temperature', 1.0) if args is not None else 1.0
    logits = logits / temperature

    return logits, labels

