from DLBio import pt_run_parallel
import subprocess


AVAILABLE_GPUS = [0] * 8


def run():
    make_object = pt_run_parallel.MakeObject(EvalBaselineProcess)
    pt_run_parallel.run(param_generator(), make_object,
                        available_gpus=AVAILABLE_GPUS,
                        do_not_check_free_gpus=True
                        )


class EvalBaselineProcess(pt_run_parallel.ITrainingProcess):
    def __init__(self, **kwargs):
        super(EvalBaselineProcess, self).__init__(**kwargs)
        self.__name__ = f'eval_baseline_process_{kwargs["baseline_thres"]}'
        self.module_name = 'run_eval_simulation.py'
        self.kwargs = kwargs


def param_generator():
    for thres in [100]:
        yield {
            'baseline_thres': thres,
            'use_baseline': None,
            'model_path': 'baseline',
        }


if __name__ == "__main__":
    run()
