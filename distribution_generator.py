import os
import random

def get_distribution(seed, size, fn):
    random.seed(seed)
    scores = [random.random() for _ in xrange(size)]
    den = sum(scores)
    probs = [score / den for score in scores]
    with open(fn, 'w') as f:
        for p in probs:
            f.write(str(p) + '\n')


def main():
    if not os.path.isdir('distributions'):
        os.makedirs('distributions')
    get_distribution(2, 3 * 6 * 40 * 32 * 32, 'distributions/one_color.list')
    get_distribution(3, 3 * 3 * 6 * 40 * 32 * 32, 'distributions/three_colors.list')


if __name__ == '__main__':
    main()
