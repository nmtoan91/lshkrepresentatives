#from Measures.Overlap import Overlap

class MeasureManager(object):
    CURRENT_MEASURE = "None"
    CURRENT_DATASET = "None"
    IS_LOAD_AUTO = False
    MEASURE_LIST = ['OF','Overlap','New_Lin','DILCA','Eskin','IOF','Nguyen_Huynh','Goodall1']
    DATASET_LIST = ['soybean_small.csv','balance-scale.csv','audiology_c.csv','zoo_c.csv','tae_c.csv','post-operative.csv','hayes-roth_c.csv','dermatology_c.csv','soybean.csv','nursery.csv','connect.csv','chess.csv','breast.csv','car.csv','mushroom.csv','splice.csv','tictactoe.csv','vote.csv','flare.csv','lymph.csv','lung.csv']
    DATASET_LIST_MK = ['soybean_small.csv','balance-scale.csv','balance-scale.csv','zoo_c.csv','tae_c.csv','post-operative.csv','hayes-roth_c.csv','dermatology_c.csv','soybean.csv','nursery.csv','connect.csv','chess.csv','breast.csv','car.csv','mushroom.csv','splice.csv','tictactoe.csv','vote.csv','flare.csv','lymph.csv','lung.csv']
    DATASET_LIST_BIG = ['big.csv']

    def Init(self):
        print("OK")

