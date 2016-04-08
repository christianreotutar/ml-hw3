from gibbs_sampler import GibbsSampler
from data import TrainData, TestData
import random, pdb, math

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

        self._c = 2     # TODO assumption: number of collections/corpuses

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
            print("iteration " + str(t))
            # go through all training documents
            print("going through training docs")
            for d in range(len(self._train)):
                line = self._train[d]
                c = int(line[0])
                tokens = line[1:]
                # go through all tokens
                print("doc " + str(d))
                for i in range(len(tokens)):
                    token = tokens[i]
                    # update counts to exclude this token
                    # TODO
                    # sample z_d_i according to Eq. 3
                    zdi_prob = self.calc_z_d_i(self._train_data, c, d, i)
                    zdi_class = self.sample(zdi_prob)
                    # sample x_d_i according to Eq. 4 using above z_d_i
                    xdi_prob = self.calc_x_d_i(self._train_data, c, d, i, zdi_class)
                    xdi_val = self.sample(xdi_prob)
                    # update counts to include this token
                    # TODO
            print("estimating training")
            # estimate theta according to Eq. 5
            self.estimate_theta(self._train_data)
            # estimate phi according to Eq. 6
            self.estimate_phi(self._train_data)
            # estimate phi(c) according to Eq. 7
            self.estimate_phi_c(self._train_data)
            # if burn period is passed
            if t > self._num_burn_in:
                # incorporate estimated params into estimate of expected val
                # TODO
                pass

            # go through all testing documents
            for d in range(len(self._test)):
                line = self._test[d]
                c = int(line[0])
                tokens = line[1:]
                # go through all tokens
                for i in range(len(tokens)):
                    token = tokens[i]
                    # update counts to exclude this token
                    # TODO
                    # sample z_d_i according to Eq. 3
                    zdi_prob = self.calc_z_d_i(self._test_data, c, d, i)
                    zdi_class = self.sample(zdi_prob)
                    # sample x_d_i according to Eq. 4 using above z_d_i
                    xdi_prob = self.calc_x_d_i(self._test_data, c, d, i, zdi_class)
                    xdi_val = self.sample(xdi_prob)
                    # update counts to include this token
                    # TODO
                    
            # compute train log-likelihood described in 3.0.1
            log_prob = self.compute_log_likelihood(self._train_data)
            # compute test log-likelihood described in 3.0.1
            log_prob = self.compute_log_likelihood(self._test_data)
        return

    '''
        Initializes values for x, z, vocab, V, and nwk map
    '''
    def initialize_values(self):

        # training
        train_vocab = set()
        train_x, train_z = [], []       # x and z values
        train_nwk_map = []              # c x k x w array for num counts

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

        # create theta, d x k
        theta = []
        for d in range(len(self._train)):
            theta.append([0.0 for _ in range(self._K)])

        # create phi, k x w
        phi = []
        for k in range(self._K):
            phi.append([0.0 for _ in range(len(train_vocab))])

        # create phi_c, c x k x w
        phi_c = []
        for c in range(self._c):
            phi_c.append([])
            for k in range(self._K):
                phi_c[c].append([0.0 for _ in range(len(train_vocab))])
            
        self._train_data = TrainData(self._train, list(train_vocab), train_x, train_z, train_nwk_map, theta, phi, phi_c)

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

        # create theta, d x k
        test_theta = []
        for d in range(len(self._train)):
            test_theta.append([0.0 for _ in range(self._K)])

        # create phi, k x w
        test_phi = []
        for k in range(self._K):
            test_phi.append([0.0 for _ in range(len(train_vocab))])

        # create phi_c, c x k x w
        test_phi_c = []
        for c in range(self._c):
            test_phi_c.append([])
            for k in range(self._K):
                test_phi_c[c].append([0.0 for _ in range(len(train_vocab))])

        self._test_data = TestData(self._test, list(test_vocab), test_x, test_z, test_nwk_map, test_theta, test_phi, test_phi_c)
        return

    '''
        According to Eq. 3 on assignment page
        p(z_d_i) prop to (ndk + a / nd* + Ka) * (nkw + b / nk* + Vb)
        @param in_data  Data object holding data for that set
        @param in_c     int corpus number
        @param in_d     int document number
        @param in_i     int token number
    '''
    def calc_z_d_i(self, in_data, in_c, in_d, in_i):
        # TODO depends on x
        # TODO need to change depending on data (i.e. use phi from train when it's test)

        prob_z_k = [0 for _ in range(self._K)]
        for k in range(self._K):
            first_term = float( in_data.get_n_d_k(in_d, k) + self._a ) / float( in_data.get_n_d_star(in_d) + ( self._K * self._a ) )

            word = in_data.get_word(in_d, in_i)
            second_term = float( in_data.get_n_k_w(k, word) + self._b ) / float( in_data.get_n_k_star(k) + ( in_data.get_V() * self._b) )
            prob_z_k[k] = first_term * second_term

        return prob_z_k

    '''
        According to Eq. 4 on assignment page
        @param data Data object holding data for that set
        @param in_c     int corpus number
        @param in_d     int document number
        @param in_i     int token number
        @param in_z_d_i int class for ith token of doc d chosen
    '''
    def calc_x_d_i(self, in_data, in_c, in_d, in_i, in_z_d_i):
        word = in_data.get_word(in_d, in_i)
        p0 = float( 1 - self._l ) * float(in_data.get_n_k_w(in_z_d_i, word) + self._b) / float( in_data.get_n_k_star(in_z_d_i) + in_data.get_V() * self._b )
        p1 = float(self._l) * float(in_data.get_n_ck_w(in_c, in_z_d_i, word) + self._b) / float( in_data.get_n_ck_star(in_c, in_z_d_i) + in_data.get_V() * self._b )
        return [p0, p1]

    '''
        Samples from a probability distribution by uniformly
        sampling from the cdf.
        @param prob_dist    List of probabilities
    '''
    def sample(self, prob_dist):
        prob_sum = sum(prob_dist)

        if (prob_sum < math.exp(-20)):
            print("Unusually small probability distribution")

        choice = random.uniform(0, prob_sum)
        running_sum = prob_sum
        for i in range(len(prob_dist)):
            if (running_sum <= choice):
                return i

            running_sum -= prob_dist[i]
        return len(prob_dist) - 1

    '''
        Estimates theta according to Eq 5
        @param in_data  Data object for all data
    '''
    def estimate_theta(self, in_data):
        for in_d in range(len(in_data.get_raw_data())):
            for in_k in range(self._K):
                num = in_data.get_n_d_k(in_d, in_k) + self._a
                denom = in_data.get_n_d_star(in_d) + (self._K * self._a)
                theta_d_k = float(num) / float(denom)
                in_data.set_theta_d_k(in_d, in_k, theta_d_k)
        
    '''
        Estimates phi according to Eq 6
        @param in_data  Data object for all data
    '''
    def estimate_phi(self, in_data):
        for in_k in range(self._K):
            for in_w in range(len(in_data.get_vocab())):
                word = in_data.get_vocab()[in_w]
                num = in_data.get_n_k_w(in_k, word) + self._b
                denom = in_data.get_n_k_star(in_k) + (in_data.get_V() * self._b)
                in_data.set_phi_k_w(in_k, in_w, float(num)/float(denom))

    '''
        Estimates phi_c according to Eq 7
        @param in_data  Data object for all data
    '''
    def estimate_phi_c(self, in_data):
        for in_c in range(self._c):
            for in_k in range(self._K):
                for in_w in range(len(in_data.get_vocab())):
                    word = in_data.get_vocab()[in_w]
                    num = in_data.get_n_ck_w(in_c, in_k, word) + self._b
                    denom = in_data.get_n_ck_star(in_c, in_k) + (in_data.get_V() * self._b)
                    in_data.set_phi_ck_w(in_c, in_k, in_w, float(num) / float(denom))

    '''
        Calculates the log likelihood according to Eq 8
    '''
    def calculate_log_likelihood(self, in_data):
        ret = 0
        for d in range(len(in_data.get_raw_data())):
            c = int(in_data.get_raw_data()[d][0])
            for i in range(len(in_data.get_raw_data()[d]) - 1):
                log_term = 0
                for z in range(self._K):
                    #TODO map i to a word idx
                    log_term += in_data.get_theta_d_k(d, k) * ((1 - self._l) * in_data.get_phi_k_w(z, i) + self._l * in_data.get_phi_ck_w(c, z, i))
                log_term = math.log(log_term)
                ret += log_term
        return ret
