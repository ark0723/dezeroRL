import numpy as np
import matplotlib.pyplot as plt
import os

# =============================================================================
# CurveGraph
# =============================================================================


class CurveGraph:
    def __init__(self, max_epoch, x_label="epoch", y_label="loss"):
        self.epochs = list(range(max_epoch))
        self.y_label = y_label
        self.x_label = x_label
        self.legend = None

    def set_legend(self, *legend):
        self.legend = legend

    def show_graph(self, *data_list, to_file=""):
        # Clear the previous plot to avoid overlap
        plt.clf()

        if self.legend is not None:
            assert len(self.legend) == len(data_list)

        for i, data in zip(self.legend, data_list):
            plt.plot(self.epochs, data, label=i)
        plt.legend(loc="upper right")
        plt.xlabel(self.x_label)
        plt.ylabel(self.y_label)

        if to_file:
            img_format = to_file.split(".").pop()
            tmp_dir = os.path.join(os.path.expanduser("~"), "./Desktop/Dezero/result")
            if not os.path.exists(tmp_dir):
                os.mkdir(tmp_dir)
            save_path = os.path.join(tmp_dir, to_file)
            plt.savefig(
                save_path,
                dpi=300,
                bbox_inches="tight",
                transparent=False,
                format=img_format,
            )

        else:
            plt.show()
