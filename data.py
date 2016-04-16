import pdb
import numpy
import sys

class Data:
    '''
        Creates a new Data instance
        @param in_raw_data list of strings, raw training data
        @param in_vocab list of unique strings
        @param in_x 2d arr of dimension (d x # tokens in d)
        @param in_z 2d arr of dimension (d x # tokens in d)
        @param in_ndk_map   (d x k) list
        @param in_nckw_map   list (dim c) of list (dim K) of hashmaps (token -> freq)
        @param in_nckw_map_star (c x K x # total tokens in K)
        @param in_theta list of lists (d x k)
        @param in_phi   list of lists (k x w)
        @param in_phi_c list of lists (k x w)
    '''
    def __init__(self, in_raw_data, in_vocab, in_x, in_z, in_ndk_map, in_nckw_map, in_nckw_map_star, in_theta, in_phi, in_phi_c, in_vocab_map):
        self._raw_data = in_raw_data
        self._vocab = in_vocab  
        self._V = len(in_vocab)
        self._x = in_x
        self._z = in_z
        self._ndk_map = in_ndk_map
        self._nckw_map = in_nckw_map
        self._nckw_map_star = in_nckw_map_star

        self._theta = in_theta
        self._phi = in_phi
        self._phi_c = in_phi_c

        self._vocab_map = in_vocab_map

    '''
        Gets the raw training/testing data
        @return list of lists
    '''
    def get_raw_data(self):
        return self._raw_data

    '''
        Returns the word in the raw data
        @param in_d The document number
        @param in_i The index number (not accounting for first c field)
        @return string
    '''
    def get_word(self, in_d, in_i):
        return self._raw_data[in_d][in_i+1]

    '''
        @return list of unique strings
    '''
    def get_vocab(self):
        return self._vocab

    '''
        @return 2d array
    '''
    def get_x(self):
        return self._x

    '''
        @param d    int document number
        @return list
    '''
    def get_x_d(self, d):

        if type(d) is not int:
            raise Exception("d is not of type int")

        return self._x[d]

    '''
        @param d    int document number
        @param i    int token number
        @return int
    '''
    def get_x_d_i(self, d, i):

        if type(d) is not int:
            raise Exception("d is not of type int")

        if type(i) is not int:
            raise Exception("i is not of type int")

        return self._x[d][i]

    '''
        @param d    int document number
        @param i    int token number
        @param val  int 0 or 1
    '''
    def set_x_d_i(self, d, i, val):

        if type(d) is not int:
            raise Exception("d is not of type int")

        if type(i) is not int:
            raise Exception("i is not of type int")

        if type(val) is not int:
            raise Exception("Val not of type int")

        self._x[d][i] = val

    '''
        @return 2d array
    '''
    def get_z(self):
        return self._z

    '''
        @param d    int document number
        @return list
    '''
    def get_z_d(self, d):

        if type(d) is not int:
            raise Exception("d is not of type int")

        return self._z[d]

    '''
        @param d    int document number
        @param i    int token number
        @return int
    '''
    def get_z_d_i(self, d, i):

        if type(d) is not int:
            raise Exception("d is not of type int")

        if type(i) is not int:
            raise Exception("i is not of type int")

        return self._z[d][i]

    '''
        @param d    int document number
        @param i    int token number
        @param val  int 0 or 1
    '''
    def set_z_d_i(self, d, i, val):

        if type(d) is not int:
            raise Exception("d is not of type int")

        if type(i) is not int:
            raise Exception("i is not of type int")

        if type(val) is not int:
            raise Exception("Val not of type int")

        self._z[d][i] = val

    '''
        @return int
    '''
    def get_V(self):
        return self._V

    '''
        Gets number of tokens in doc d assigned to class k
        @return int number of times tokens in d are assigned k
        @param d    int document number
        @param k    int class
    '''
    def get_n_d_k(self, d, k):

        if type(d) is not int:
            raise Exception("d not of type int")

        if type(k) is not int:
            raise Exception("k not of type int")

        return self._ndk_map[d][k]

    '''
        Gets number of tokens in doc d assigned to any class
        @param in_d    int the document number
        @return     int count of the number
    '''
    def get_n_d_star(self, in_d):

        if type(in_d) is not int:
            raise Exception("d not of type int")

        count = len(self._z[in_d])
        return count

    '''
        Get the number of tokens of type w assigned to k
        @param in_k int class of the token we're matching
        @param in_w_idx int word of the token we're matching
    '''
    def get_n_k_w(self, in_k, in_w_idx):

        num = 0
        for c in range(len(self._nckw_map)):
            num += self.get_n_ck_w(c, in_k, in_w_idx)

        return num

    '''
        Get the number of tokens of all types assigned to k
        @param in_k int class of the token we're matching
    '''
    def get_n_k_star(self, in_k):

        num = 0
        for c in range(len(self._nckw_map)):
            num += self.get_n_ck_star(c, in_k)
        
        return num

    '''
        Get the number of tokens of type w assigned to k
        @param in_c int corpus to check
        @param in_k int class of the token we're matching
        @param in_w_idx index of the word of the token we're matching
    '''
    def get_n_ck_w(self, in_c, in_k, in_w_idx):
        if (in_c >= len(self._nckw_map)):
            pdb.set_trace()
            raise Exception("incorrect index c: " + str(in_c))

        if (in_k >= len(self._nckw_map[in_c])):
            raise Exception("incorrect index k: " + str(in_k))

        if (in_w_idx >= len(self._nckw_map[in_c][in_k])):
            # should this happen?
            return 0

        return self._nckw_map[in_c][in_k][in_w_idx]


    '''
        Get the number of tokens of all types assigned to k
        @param in_c int corpus to check
        @param in_k int class of the token we're matching
    '''
    def get_n_ck_star(self, in_c, in_k):

        if (in_c >= len(self._nckw_map)):
            raise Exception("incorrect index c: " + str(in_c))
        
        if (in_k >= len(self._nckw_map[in_c])):
            raise Exception("incorrect index k: " + str(in_k))

        return self._nckw_map_star[in_c][in_k]

    '''
        Sets the theta
        @param in_d     int document number
        @param in_k     int class number
        @param in_theta float theta val
    '''
    def set_theta_d_k(self, in_d, in_k, in_theta):
        #TODO check d
        #TODO check k
        self._theta[in_d][in_k] = in_theta

    '''
        Gets the theta
        @param in_d     int document number
        @param in_k     int class number
        @return         float theta val
    '''
    def get_theta_d_k(self, in_d, in_k):
        return self._theta[in_d][in_k]

    '''
        Sets the Phi
        @param in_k     int class number
        @param in_w     int word idx
        @param in_phi   float phi val
    '''
    def set_phi_k_w(self, in_k, in_w, in_phi):
        #TODO check k
        #TODO check w
        self._phi[in_k][in_w] = in_phi

    '''
        Gets the Phi
        @param in_k     int class number
        @param in_w     int word idx
        @return         float phi value
    '''
    def get_phi_k_w(self, in_k, in_w):
        #TODO check k
        #TODO check w
        return self._phi[in_k][in_w]

    '''
        Sets the Phi(c)
        @param in_c     int corpus
        @param in_k     int class number
        @param in_w     int word idx
        @param in_phi   float phi val
    '''
    def set_phi_ck_w(self, in_c, in_k, in_w, in_phi):
        #TODO check c
        #TODO check k
        #TODO check w
        self._phi_c[in_c][in_k][in_w] = in_phi

    '''
        Gets the Phi(c)
        @param in_c     int corpus
        @param in_k     int class number
        @param in_w     int word idx
        @return         float phi value
    '''
    def get_phi_ck_w(self, in_c, in_k, in_w):
        #TODO check c
        #TODO check k
        #TODO check w
        return self._phi_c[in_c][in_k][in_w]

    def set_theta(self, in_theta):
        self._theta = in_theta

    def set_phi(self, in_phi):
        self._phi = in_phi

    def set_phi_c(self, in_phi_c):
        self._phi_c = in_phi_c

    def exclude_token(self, in_c, in_d, in_i, in_token):
        in_z = self._z[in_d][in_i]
        in_x = self._x[in_d][in_i]

        token_idx = self._vocab_map[in_token]

        if (self._ndk_map[in_d][in_z] - 1 < 0):
            pdb.set_trace()

        self._ndk_map[in_d][in_z] = self._ndk_map[in_d][in_z] - 1
        self._nckw_map[in_c][in_z][token_idx] = self._nckw_map[in_c][in_z][token_idx] - 1
        self._nckw_map_star[in_c][in_z] = self._nckw_map_star[in_c][in_z] - 1

    def include_token(self, in_c, in_d, in_i, in_token, in_z, in_x):

        token_idx = self._vocab_map[in_token]

        self._ndk_map[in_d][in_z] = self._ndk_map[in_d][in_z] + 1
        self._nckw_map[in_c][in_z][token_idx] = self._nckw_map[in_c][in_z][token_idx] + 1
        self._nckw_map_star[in_c][in_z] = self._nckw_map_star[in_c][in_z] + 1

    '''
        @param word     string
        @return int index of vocab word
    '''
    def get_word_idx(self, in_word):
        return self._vocab_map[in_word]

    def __str__(self):
        string = "\n"
        string += "\nRAW DATA:\n" + str(self._raw_data)
        string += "\nVOCAB:\n" + str(self._vocab)
        string += "\nV:\n" + str(self._V)
        string += "\nX:\n" + str(self._x)
        string += "\nZ:\n" + str(self._z)
        string += "\nNDK MAP:\n" + str(self._ndk_map)
        string += "\nNCKW MAP:\n" + str(self._nckw_map)
        string += "\nNCKW MAP STAR\n" + str(self._nckw_map_star)

        string += "\nTHETA:\n" + str(self._theta)
        string += "\nPHI:\n" + str(self._phi)
        string += "\nPHI_C\n" + str(self._phi_c)

        string += "\nVOCAB MAP:\n" + str(self._vocab_map)
        return string

    def __repr__(self):
        return self.__str__()