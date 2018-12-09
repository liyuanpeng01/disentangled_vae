import numpy as np

def get_one_recall(a, b_list):
    best_v = -1
    best_i = -1
    for i, b in enumerate(b_list):
        if np.var(b) > 0:
            cur = abs(np.corrcoef(a, b)[0][1])
            if cur > best_v:
                best_v = cur
                best_i = i
    return best_i, best_v


def get_ave_recall(a_list, b_list):
    cor_list = []
    all = 0.
    for a in a_list:
        index, value = get_one_recall(a, b_list)
        cor_list.append((index, value))
        all += value
    return all / len(a_list), cor_list


if __name__ == '__main__':
    a = [1., 0, 2.3, 1.3, 2.9]
    b = [0, 0, 0, 0, 1]

    print np.var(a)
    print np.var(b)
    print np.corrcoef(a, b)
