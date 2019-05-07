from dropout_factory.dropout import Dropout


class DropoutFactory(object):
    def __init__(self):
        self.__instances = dict()

    def get_instance (self, size_a1, size_a2, dropout_rate):
        if (size_a1, size_a2 , dropout_rate) not in self.__instances:
            self.__instances[(size_a1,size_a2,dropout_rate)] = Dropout(size_a1,size_a2,dropout_rate)
        return self.__instances[(size_a1,size_a2,dropout_rate)]
