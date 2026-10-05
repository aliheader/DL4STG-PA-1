import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display

from data import RESULTS_DIR

OUTPUT_DIR = RESULTS_DIR / "outputs"


def publish(number, table=None, directory=OUTPUT_DIR):
    directory.mkdir(parents=True, exist_ok=True)
    print(f"Output {number}")
    if table is not None:
        table = table.copy()
        table.to_csv(directory / f"{number}.csv", index=False)
        table.to_latex(
            directory / f"{number}.tex",
            index=False,
            escape=True,
            float_format="%.2f",
            na_rep="--",
        )
        display(table.round(2).replace({np.nan: "--"}))
    for index, figure_id in enumerate(plt.get_fignums(), 1):
        figure = plt.figure(figure_id)
        figure.savefig(directory / f"{number}-{index}.pdf", bbox_inches="tight")
        display(figure)
        plt.close(figure)
