# show_result

import matplotlib.pyplot as plt

def show_result(tab_train, tab_test):
    plt.ylim(0, 1)
    plt.grid()
    plt.plot(tab_train, label="Train error")
    plt.plot(tab_test, label="Test error")
    plt.legend(loc="upper right")
    plt.show()
    pass
