from gibbs_sampler import GibbsSampler
from data import TrainData, TestData
import random, pdb

'''
    Collapsed Gibbs Sampler
'''
class CollapsedSampler(GibbsSampler):

    '''
        Creates a Collapsed Sampler
        @param in_train list of lists of [classification words]
        @param in_test line of lists of [classification words]
        @param in_out   string path for output
        @param in_k int number of topics
        @param in_l float lambda for c variable
        @param in_a float alpha for theta variable
        @param in_b float beta for phi variable
        @param in_num_iters int number of total iterations for algorithm
        @param in_num_burn_in   int number of "burn in" iterations
    '''
    def __init__(self, in_train, in_test, in_out, in_k, in_l, in_a,\
                        in_b, in_num_iters, in_num_burn_in):
        self._train = in_train
        self._test = in_test
        self._K = in_k
        self._l = in_l
        self._a = in_a
        self._b = in_b
        self._num_iters = in_num_iters
        self._num_burn_in = in_num_burn_in

        self._c = 2     # number of collections/corpuses

        self._train_data = None
        self._test_data = None

    '''
        Runs the Gibbs Sampling Algorithm
    '''
    def algorithm(self):
        # set all z and x values to random in {0,...,K-1} and {0,1}
        # one per token
        print("initializing values")
        self.initialize_values()

        print("iterating")
        # go through T iterations
        for t in range(1, self._num_iters + 1):
            # go through all training documents
            for d in range(len(self._train)):
                line = self._train[d]
                c = line[0]
                tokens = line[1:]
                # go through all words
                for i in range(len(tokens)):
                    token = tokens[i]
                    # update counts to exclude this token
                    # TODO
                    # sample z_d_i according to Eq. 3
                    self.calc_z_d_i(self._train_data, d, i)
                    # sample x_d_i according to Eq. 4 using above z_d_i
                    # update counts to include this token
            # estimate theta according to Eq. 5
            # estimate phi according to Eq. 6
            # estimate phi(c) according to Eq. 7
            # if burn period is passed
            if t > self._num_burn_in:
                # incorporate estimated params into estimate of expected val
                pass

            # go through all testing documents
            for d in range(len(self._test)):
                line = self._test[d]
                c = line[0]
                tokens = line[1:]
                # go through all words
                for i in range(len(tokens)):
                    token = tokens[i]
                    # update counts to exclude this token
                    # sample z_d_i according to Eq. 3
                    # sample x_d_i according to Eq. 4 using above z_d_i
                    # update counts to include this token
                    pass
                    
            # compute train log-likelihood described in 3.0.1
            # compute test log-likelihood described in 3.0.1
        return

    '''
        Initializes values for x, z, vocab, V, and nwk map
    '''
    def initialize_values(self):

        # training
        train_vocab = set()
        train_x, train_z = [], []
        train_nwk_map = []

        # for all corpuses
        for c in range(self._c):
            train_nwk_map.append([])
            # for all topics
            for k in range(self._K):
                train_nwk_map[c].append({})

        for line in self._train:
            c = int(line[0])
            tokens = line[1:]
            doc_x = []
            doc_z = []
            for token in tokens:
                train_vocab.add(token)
                doc_x.append(random.randint(0, 1))
                z = random.randint(0, self._K - 1)
                doc_z.append(z)

                if (token not in train_nwk_map[c][z]):
                    train_nwk_map[c][z][token] = 0
                train_nwk_map[c][z][token] = train_nwk_map[c][z][token] + 1

            train_x.append(doc_x)
            train_z.append(doc_z)
            
        self._train_data = TrainData(train_vocab, train_x, train_z, train_nwk_map)

        # testing
        test_vocab = set()
        test_x, test_z = [], []
        test_nwk_map = []

        for c in range(self._c):
            test_nwk_map.append([])
            for k in range(self._K):
                test_nwk_map[c].append({})

        for line in self._test:
            c = int(line[0])
            tokens = line[1:]
            doc_x = []
            doc_z = []
            for token in tokens:
                test_vocab.add(token)
                test_x.append(random.randint(0, 1))
                z = random.randint(0, self._K - 1)
                test_z.append(z)

                if (token not in test_nwk_map[c][z]):
                    test_nwk_map[c][z][token] = 0
                test_nwk_map[c][z][token] = test_nwk_map[c][z][token] + 1

            test_x.append(doc_x)
            test_z.append(doc_z)

        self._test_data = TestData(test_vocab, test_x, test_z, test_nwk_map)
        return

    '''
        According to Eq. 3 on assignment page
        p(z_d_i) prop to (ndk + a / nd* + Ka) * (nkw + b / nk* + Vb)
        @param in_data Data object holding data for that set
        @param in_d    int document number
        @param in_i    int token number
    '''
    def calc_z_d_i(self, in_data, in_d, in_i):
        # TODO depends on x

        prob_z_k = [0 for _ in range(self._K)]
        for k in range(self._K):
            first_term = float( in_data.get_n_d_k(in_d, k) + self._a ) / float( in_data.get_n_d_star(in_d) + ( self._K * self._a ) )

            word = self._train[in_d][in_i + 1] #TODO only works train
            second_term = float( in_data.get_n_k_w(k, word) + self._b ) / float( in_data.get_n_k_star(k) + ( in_data.get_V() * self._b) )
            prob_z_k[k] = first_term * second_term

        pdb.set_trace()
        return 

    '''
        According to Eq. 4 on assignment page
        @param data Data object holding data for that set
        @param d    int document number
        @param i    int token number
    '''
    def calc_x_d_i(self, in_data, in_d, in_i, in_z_d_i):
        #word = in_data.vo
        p0 = float( 1 - self._lambda ) * float(in_data.get_n_k_w(in_z_d_i, word) + self._beta) / float( data.get_n_k_w + data.get_V() * self._beta )
        p1 = float(self._lambda) * float(data.get_c_k_w + self._beta) / float( data.get_n_c_k_star + data.get_V() * self._beta )
        return

    '''
        Samples from a probability distribution by uniformly
        sampling from the cdf.
        @param prob_dist    List of probabilities
    '''
    def sample(prob_dist):
        prob_sum = sum(prob_dist)

        if (prob_sum == 0):
            print("Unusually small probability distribution")

        choice = random.uniform(0, prob_sum)
        running_sum = prob_sum
        for i in range(len(prob_dist)):
            if (running_sum <= choice):
                return i

            running_sum -= prob_dist[i]
