from DLBio import pt_run_parallel
import copy
import argparse

AVAILABLE_GPUS = [
    0,
    1,
    2,
    3
]

USED_MODELS = [
    'smp_resnet18',
    # 'smp_resnet50',
    # 'smp_resnet152',
    # 'smp_mobilenet_v2'
]

DEFAULT_KWARGS = {
    'use_rgb': None,  # translates to use_rgb = True
    'dataset': 'simulation',
    'seed': 0
}

POST_FIX = 'sim_15032021'


def get_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', type=str, default='models')
    return parser.parse_args()


def run():
    options = get_options()
    if options.type == 'models':
        param_generator = different_models_pg
    else:
        raise ValueError(f'unknown type {options.type}')

    make_object = pt_run_parallel.MakeObject(TrainingProcess)
    pt_run_parallel.run(param_generator(), make_object,
                        available_gpus=AVAILABLE_GPUS
                        )


class TrainingProcess(pt_run_parallel.ITrainingProcess):
    def __init__(self, **kwargs):
        super(TrainingProcess, self).__init__(**kwargs)
        self.__name__ = f'train_model_{kwargs["model_type"]}'
        self.module_name = 'run_training.py'
        self.kwargs = kwargs


# -----------------------------------------------------------------------------
# ---------------------PARAM GENERATORS----------------------------------------
# -----------------------------------------------------------------------------


def different_models_pg():
    for model_type in USED_MODELS:
        kwargs = copy.deepcopy(DEFAULT_KWARGS)
        kwargs.update({
            'model_type': model_type,
            'folder': model_type + '_' + POST_FIX,
        })

        yield kwargs


if __name__ == "__main__":
    run()
