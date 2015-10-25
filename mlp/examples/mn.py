#!/usr/bin/env python
from mlp import MLP, Layer
from mnist.mnist import MNIST

DEBUG = False

def main():
    imres = MLP()
    num_points = 784
    imres.add_layer(Layer(num_points))
    imres.add_layer(Layer(20))
    imres.add_layer(Layer(10))

    imres.add_bias()
    imres.init_network()

    imres.step = 0.001
    imres.moment = imres.step / 10
    imres.verbose = True
    target_error = 0.01

    imres.patterns = []
    imres._patterns = []
    imres.test_patterns = []
    imres._test_patterns = []

    def norm(inp):
        def fn(x):
            return x/255
        return map(fn, inp)

    mn = MNIST('./mnist/data/')
    samples, labels = mn.load_testing()
    for i in range(100):
        outvect = [0]*10
        outvect[labels[i]] = 1
        imres.patterns.append((samples[i], outvect))
        imres._patterns.append((samples[i], labels[i], outvect))

    for i in range(100, 200):
        outvect = [0]*10
        outvect[labels[i]] = 1
        imres.test_patterns.append((samples[i], outvect))
        imres._test_patterns.append((samples[i], labels[i], outvect))

    print 'Training samples: %d' %  len(imres.patterns)
    print 'Testing samples: %d' %  len(imres.test_patterns)
    print 'Target error: %.4f' % target_error

    final_err, steps = imres.train_target(imres.patterns,
        target_error=target_error)

    print 'Training done in %d steps with final error of %.6f' % (steps,
        final_err)


    print '----- Detailed test output -----'
    total_tests = len(imres._test_patterns)
    total_fails = 0
    for inp, num, target in imres._test_patterns:
        computed = imres.run(inp)
        error = abs(computed[0] - target[0])
        computed = map(lambda x: round(x, 4), computed)
        maxn = computed[0]
        pos = 0
        for i in range(len(computed)):
            if computed[i] > maxn:
                maxn = computed[i]
                pos = i

        if num != pos:
            total_fails += 1
        print 'in: %d, out: %d' % (num, pos)
        print 'target: %s \noutput: %s' % (target, computed)

    print '-----'
    print 'Testing done - %d of %d samples classified incorrectly' % (
        total_fails, total_tests)


if __name__ == "__main__":
    main()
