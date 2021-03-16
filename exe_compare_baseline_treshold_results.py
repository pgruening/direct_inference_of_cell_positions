import pandas as pd
from DLBio.helpers import search_rgx, MyDataFrame
import config
from os.path import join


def run():
    files_ = search_rgx(r'results_baseline_t(\d+).xlsx', config.SIM_EVAL_BASE)
    assert files_

    df = MyDataFrame()

    for file in files_:
        tmp = pd.read_excel(
            join(config.SIM_EVAL_BASE, file), engine='openpyxl'
        )
        tmp = dict(tmp.mean())
        tmp['name'] = file
        df.update(tmp)

    df = df.get_df()
    print(df.loc[:, ['name', 'f1_score', 'dice']])


if __name__ == '__main__':
    run()
