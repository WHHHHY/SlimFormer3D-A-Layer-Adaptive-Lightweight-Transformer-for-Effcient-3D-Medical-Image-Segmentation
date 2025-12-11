from .abdomen_tester import AbdomenTester
from .btcv_tester import BTCVTester

def get_tester(opt, model, metrics=None):
    if opt["dataset_name"] == "Abdomen" :
        tester = AbdomenTester(opt, model, metrics)
    elif opt["dataset_name"] == "BTCV":
        tester = BTCVTester(opt, model, metrics)

    else:
        raise RuntimeError(f"No {opt['dataset_name']} dataset available when initialize tester")

    return tester
