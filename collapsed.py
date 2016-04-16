from gibbs_sampler import GibbsSampler
from data import Data
import random, pdb, math, sys

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
                #print("doc " + str(d))
                #print("BEFORE WE LOOP")
                #print(self._train_data)
                for i in range(len(tokens)):
                    token = tokens[i]
                    # update counts to exclude this token
                    self._train_data.exclude_token(c, d, i, token) # CORRECT
                    #print("AFTER EXLUDING")
                    #print(self._train_data)
                    # sample z_d_i according to Eq. 3
                    curr_x_d_i = self._train_data.get_x_d_i(d, i) # CORRECT
                    zdi_prob = self.calc_z_d_i(self._train_data, c, d, token, curr_x_d_i) # CORRECT
                    zdi_class = self.sample(zdi_prob) # CORRECT
                    self._train_data.set_z_d_i(d, i, zdi_class) # CORRECT
                    # sample x_d_i according to Eq. 4 using above z_d_i
                    xdi_prob = self.calc_x_d_i(self._train_data, c, d, token, zdi_class) # CORRECT
                    xdi_val = self.sample(xdi_prob) # CORRECT
                    self._train_data.set_x_d_i(d, i, xdi_val) # CORRECT
                    # update counts to include this token
                    self._train_data.include_token(c, d, i, token, zdi_class, xdi_val) 
                    #print("AFTER INCLUDING")
                    #print(self._train_data)


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


            print("going through test")
            # go through all testing documents
            for d in range(len(self._test)):
                line = self._test[d]
                c = int(line[0])
                tokens = line[1:]
                # go through all tokens
                for i in range(len(tokens)):
                    token = tokens[i]
                    # update counts to exclude this token
                    self._test_data.exclude_token(c, d, i, token)
                    # sample z_d_i according to Eq. 3
                    curr_x_d_i = self._test_data.get_x_d_i(d, i)
                    zdi_prob = self.calc_z_d_i_test(self._test_data, c, d, token, i, curr_x_d_i)
                    zdi_class = self.sample(zdi_prob)
                    self._test_data.set_z_d_i(d, i, zdi_class)
                    # sample x_d_i according to Eq. 4 using above z_d_i
                    xdi_prob = self.calc_x_d_i_test(self._test_data, c, d, token, i, zdi_class)
                    xdi_val = self.sample(xdi_prob)
                    self._test_data.set_x_d_i(d, i, xdi_val)
                    # update counts to include this token
                    self._test_data.include_token(c, d, i, token, zdi_class, xdi_val)

            print ("estimating theta param for test")
            self.estimate_theta(self._test_data)
            self._test_data.set_phi(self._train_data._phi)
            self._test_data.set_phi_c(self._train_data._phi_c)
                    
            # compute train log-likelihood described in 3.0.1
            train_log_prob = self.compute_log_likelihood(self._train_data)
            print(train_log_prob)
            # compute test log-likelihood described in 3.0.1
            test_log_prob = self.compute_log_likelihood(self._test_data)
            print(test_log_prob)

        return

    '''
        Initializes values for x, z, vocab, V, and nwk map
    '''
    def initialize_values(self):

        raw_datas = [self._train, self._test]
        datas = []

        for raw_data in raw_datas:
            _vocab = set()
            _vocab_map = {}
            _vocab_map_idx = 0
            _x, _z = [], []         # x and z values
            _nwk_map = []           # c x k x w array for num counts
            _nwk_map_star = []      # c x k x i array for num counts !!
            _ndk_map = []           # d x k array for num counts

            # for all corpuses
            for c in range(self._c):
                _nwk_map.append([])
                _nwk_map_star.append([]) # !!
                # for all topics
                for k in range(self._K):
                    _nwk_map[c].append({})
                    _nwk_map_star[c].append(0) # !!

            # for every line in the raw data
            i = -1
            for line in raw_data:
                i += 1
                c = int(line[0])
                tokens = line[1:]
                doc_x = []
                doc_z = []

                _ndk_map.append([0 for _ in range(self._K)])

                for token in tokens:

                    if token not in _vocab:
                        _vocab_map[token] = _vocab_map_idx
                        _vocab_map_idx += 1
                        _vocab.add(token)

                    doc_x.append(random.randint(0, 1))
                    z = random.randint(0, self._K - 1)
                    doc_z.append(z)

                    # update ndk map
                    _ndk_map[i][z] = _ndk_map[i][z] + 1

                    # update nwk map
                    if (token not in _nwk_map[c][z]):
                        _nwk_map[c][z][token] = 0
                    _nwk_map[c][z][token] = _nwk_map[c][z][token] + 1

                    # update nwk_map_star
                    _nwk_map_star[c][z] = _nwk_map_star[c][z] + 1 # !!

                _x.append(doc_x)
                _z.append(doc_z)

            # create theta, d x k
            theta = []
            for d in range(len(raw_data)):
                theta.append([0.0 for _ in range(self._K)])

            # create phi, k x w
            phi = []
            for k in range(self._K):
                phi.append([0.0 for _ in range(len(_vocab))])

            # create phi_c, c x k x w
            phi_c = []
            for c in range(self._c):
                phi_c.append([])
                for k in range(self._K):
                    phi_c[c].append([0.0 for _ in range(len(_vocab))])
                
            _data = Data(raw_data, list(_vocab), _x, _z, _ndk_map, _nwk_map, _nwk_map_star, theta, phi, phi_c, _vocab_map) # !!
            datas.append(_data)
        self._train_data = datas[0]
        self._test_data = datas[1]

        return

    '''
        According to Eq. 3 on assignment page
        p(z_d_i) prop to (ndk + a / nd* + Ka) * (nkw + b / nk* + Vb)
        @param in_data  Data object holding data for that set
        @param in_c     int corpus number
        @param in_d     int document number
        @param in_token     string token
        @param in_x_d_i     int whether we use corpus or global counts
    '''
    def calc_z_d_i(self, in_data, in_c, in_d, in_token, in_x_d_i):
        # TODO need to change depending on data (i.e. use phi from train when it's test)
        
        prob_z_k = [0 for _ in range(self._K)]
        word = in_token
        # USE GLOBAL COUNTS
        if (in_x_d_i == 0):
            for k in range(self._K):
                first_term = float( in_data.get_n_d_k(in_d, k) + self._a ) / float( in_data.get_n_d_star(in_d) + ( self._K * self._a ) )
                second_term = float( in_data.get_n_k_w(k, word) + self._b ) / float( in_data.get_n_k_star(k) + ( in_data.get_V() * self._b) )
                prob_z_k[k] = first_term * second_term
        # USE CORPUS SPECIFIC COUNTS
        elif (in_x_d_i == 1):
            for k in range(self._K):
                first_term = float( in_data.get_n_d_k(in_d, k) + self._a ) / float( in_data.get_n_d_star(in_d) + ( self._K * self._a ) )
                second_term = float( in_data.get_n_ck_w(in_c, k, word) + self._b ) / float( in_data.get_n_ck_star(in_c, k) + ( in_data.get_V() * self._b) )
                prob_z_k[k] = first_term * second_term

        return prob_z_k


    '''
        According to Eq. 3 on assignment page EXCEPT WITH PHI REPLACED FOR THE TEST DATA
        p(z_d_i) prop to (ndk + a / nd* + Ka) * (nkw + b / nk* + Vb)
        @param in_data  Data object holding data for that set
        @param in_c     int corpus number
        @param in_d     int document number
        @param in_token     string token
        @param in_i         ith iteration of dth document
        @param in_x_d_i     int whether we use corpus or global counts
    '''
    def calc_z_d_i_test(self, in_data, in_c, in_d, in_token, in_i, in_x_d_i):
        # TODO need to change depending on data (i.e. use phi from train when it's test)
        
        prob_z_k = [0 for _ in range(self._K)]
        word = in_token
        word_idx = in_data.get_word_idx(word)
        # USE GLOBAL COUNTS
        if (in_x_d_i == 0):
            for k in range(self._K):
                first_term = float( in_data.get_n_d_k(in_d, k) + self._a ) / float( in_data.get_n_d_star(in_d) + ( self._K * self._a ) )
                second_term = in_data.get_phi_k_w(k, word_idx)
                prob_z_k[k] = first_term * second_term
        # USE CORPUS SPECIFIC COUNTS
        elif (in_x_d_i == 1):
            for k in range(self._K):
                first_term = float( in_data.get_n_d_k(in_d, k) + self._a ) / float( in_data.get_n_d_star(in_d) + ( self._K * self._a ) )

                second_term = in_data.get_phi_ck_w(in_c, k, word_idx)
                prob_z_k[k] = first_term * second_term

        return prob_z_k

    '''
        According to Eq. 4 on assignment page
        @param data Data object holding data for that set
        @param in_c     int corpus number
        @param in_d     int document number
        @param in_token     string token
        @param in_z_d_i int class for ith token of doc d chosen
    '''
    def calc_x_d_i(self, in_data, in_c, in_d, in_token, in_z_d_i):
        word = in_token
        p0 = float( 1 - self._l ) * float(in_data.get_n_k_w(in_z_d_i, word) + self._b) / float( in_data.get_n_k_star(in_z_d_i) + in_data.get_V() * self._b )
        p1 = float(self._l) * float(in_data.get_n_ck_w(in_c, in_z_d_i, word) + self._b) / float( in_data.get_n_ck_star(in_c, in_z_d_i) + in_data.get_V() * self._b )
        return [p0, p1]


    '''
        According to Eq. 4 on assignment page EXCEPT PHI IS REPLACED FOR THE TEST
        @param data Data object holding data for that set
        @param in_c     int corpus number
        @param in_d     int document number
        @param in_token     string token
        @param in_i         ith iteration of dth document
        @param in_z_d_i int class for ith token of doc d chosen
    '''
    def calc_x_d_i_test(self, in_data, in_c, in_d, in_token, in_i, in_z_d_i):
        word = in_token
        word_idx = in_data.get_word_idx(word)
        p0 = float( 1 - self._l ) * in_data.get_phi_k_w(in_z_d_i, word_idx)
        p1 = float(self._l) * in_data.get_phi_ck_w(in_c, in_z_d_i, word_idx)
        return [p0, p1]

    '''
        Samples from a probability distribution by uniformly
        sampling from the cdf.
        @param prob_dist    List of probabilities
        @return index
    '''
    def sample(self, prob_dist):
        prob_sum = sum(prob_dist)

        #if (prob_sum < math.exp(-20)):
        #    print("Unusually small probability distribution")

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
                if (theta_d_k < 0):
                    print("negative theta")
                    sys.exit(1)
        
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
    def compute_log_likelihood(self, in_data):
        ret = 0
        for d in range(len(in_data.get_raw_data())):
            c = int(in_data.get_raw_data()[d][0])
            for i in range(1, len(in_data.get_raw_data()[d])):
                word = in_data.get_raw_data()[d][i]
                log_term = 0
                for z in range(self._K):
                    #... this is wrong
                    word_idx = in_data.get_word_idx(word)
                    log_term += in_data.get_theta_d_k(d, z) * ((1 - self._l) * in_data.get_phi_k_w(z, word_idx) + self._l * in_data.get_phi_ck_w(c, z, word_idx))
                    if (log_term <= 0):
                        print("negative log term")
                        sys.exit(1)
                if (log_term == 0):
                    pdb.set_trace()
                log_term = math.log(log_term)
                ret += log_term
        return ret
