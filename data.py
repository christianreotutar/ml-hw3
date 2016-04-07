class Data:
    def __init__(self):
        raise NotImplementedError

    '''
        @return set
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

        self._x[d][i] = val

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

        # TODO optimize
        count = 0
        for class_desig in self._z[d]:
            if (class_desig == k):
                count += 1
        return count

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
        @param in_w string word of the token we're matching
    '''
    def get_n_k_w(self, in_k, in_w):

        num = 0
        for c in range(len(self._nwk_map)):
            num += self.get_n_ck_w(c, in_k, in_w)

        return num

    '''
        Get the number of tokens of all types assigned to k
        @param in_k int class of the token we're matching
    '''
    def get_n_k_star(self, in_k):

        num = 0
        for c in range(len(self._nwk_map)):
            num += self.get_n_ck_star(c, in_k)
        
        return num

    '''
        Get the number of tokens of type w assigned to k
        @param in_c int corpus to check
        @param in_k int class of the token we're matching
        @param in_w string word of the token we're matching
    '''
    def get_n_ck_w(self, in_c, in_k, in_w):

        if (in_c >= len(self._nwk_map)):
            raise Exception("incorrect index c: " + str(in_c))

        if (in_k >= len(self._nwk_map[in_c])):
            raise Exception("incorrect index k: " + str(in_k))

        if (in_w not in self._nwk_map[in_c][in_k]):
            # should this happen?
            return 0

        return self._nwk_map[in_c][in_k][in_w]

    '''
        Get the number of tokens of all types assigned to k
        @param in_c int corpus to check
        @param in_k int class of the token we're matching
    '''
    def get_n_ck_star(self, in_c, in_k):

        if (in_c >= len(self._nwk_map)):
            raise Exception("incorrect index c: " + str(in_c))
        
        if (in_k >= len(self._nwk_map[in_c])):
            raise Exception("incorrect index k: " + str(in_k))

        return sum(self._nwk_map[in_c][in_k].values())

'''
    Subclass of Data
    The training data class
    Different from TestData in that phi is always updated
'''
class TrainData(Data):

    '''
        Creates a new TrainData instance
        @param in_vocab set of strings
        @param in_x 2d arr of dimension (d x # tokens in d)
        @param in_z 2d arr of dimension (d x # tokens in d)
        @param in_nwk_map   list (dim c) of list (dim K) of hashmaps (token -> freq)
    '''
    def __init__(self, in_vocab, in_x, in_z, in_nwk_map):
        self._vocab = in_vocab  
        self._V = len(in_vocab)
        self._x = in_x
        self._z = in_z
        self._nwk_map = in_nwk_map

'''
    Subclass of Data
    The testing data class
    Different from TrainingData in that phi is used from training data
'''
class TestData(Data):
    def __init__(self, in_vocab, in_x, in_z, in_nwk_map):
        self._vocab = in_vocab
        self._V = len(in_vocab)
        self._x = in_x
        self._z = in_z
        self._nwk_map = in_nwk_map
