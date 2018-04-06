import matplotlib.pyplot as plt

def plot(train_acc, test_acc, time_diff, pic_path = None):

    epoch = len(train_acc)
    l1, = plt.plot(range(epoch), train_acc, 'r', label = 'train')
    l2, = plt.plot(range(epoch), test_acc, 'b', label = 'test')
    plt.legend(handles = [l1, l2])
    plt.title('Accuracy for time ' + time_diff + 'delay')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    if pic_path is not None:
        plt.savefig(pic_path)
    else:
        plt.show()