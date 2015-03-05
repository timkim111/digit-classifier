import numpy as np
import heapq

TEST1 = 0
TEST2 = 1

testFeatures = open('testFeatures.csv', 'r')
trainFeatures = open('trainFeatures.csv', 'r')
trainLabels = open('trainLabels.csv', 'r')
valFeatures = open('valFeatures.csv', 'r')
valLabels = open('valLabels.csv', 'r')

def list_str_to_float(lst):
    res = []
    for i in lst:
        res.append(float(i))
    return res

def classify(k, write_file, F_L, validation):
    write_to = open(write_file, 'w')
    def k_insert(kh, dist, pair):
        if len(kh) < k:
            heapq.heappush(kh, (-dist, pair))
        else:
            heapq.heappush(kh, (-dist, pair))
            heapq.heappop(kh)

    res = []
    progress_index = 0
    hits = 0

    for v_pair in F_L:
        v = v_pair.f
        class_v = v_pair.l
        k_heap = []
        for t_pair in t_F_L:
            t = t_pair.f
            class_t = t_pair.l
            euclid_dist = ((v - t) ** 2).sum() ** .5
            k_insert(k_heap, euclid_dist, t_pair)
        if TEST1:
            print("Running classify " + str(k) + ": " + str(progress_index) + "/" + str(len(F_L)))
            print(k_heap)
            print("")
        k_closest = [(i[0], i[1]) for i in k_heap]
        occurances = [[0, 0] for i in range(10)] #index = num, value = (times in k_closest, total distance in k_closest)
        for i in k_closest:
            occurances[i[1].l][0] += 1
            occurances[i[1].l][1] += i[0]
        freq_dist_nums = [(occurances[i][0], occurances[i][1], i) for i in range(len(occurances))]
        best_choice = max(freq_dist_nums)
        write_to.write(str(best_choice[2]) + '\n')
        if validation and (best_choice[2] == F_L[progress_index].l):
            hits += 1
        progress_index += 1
    write_to.close()
    if validation:
        if TEST2:
            print("k = " + str(k) + ": " + str(float(hits) / len(F_L)))
        return float(hits) / len(F_L)

tf = trainFeatures.readlines()
tf = [list_str_to_float(i.strip().split(',')) for i in tf]
tf = [np.array(i) for i in tf]
tl = trainLabels.readlines()
tl = [int(i.strip()) for i in tl]

vf = valFeatures.readlines()
vf = [list_str_to_float(i.strip().split(',')) for i in vf]
vf = [np.array(i) for i in vf]
vl = valLabels.readlines()
vl = [int(i.strip()) for i in vl]

testf = testFeatures.readlines()
testf = [list_str_to_float(i.strip().split(',')) for i in testf]
testf = [np.array(i) for i in testf]
testl = [None for i in range(len(testf))]

#feature_value pair
class F_L_Pair:
    def __init__(self, f, l):
        self.f = f
        self.l = l
#   def __repr__(self):
#       return "P(" + repr(self.f) + ", " + repr(self.l) + ")"

t_F_L = []
for i in range(len(tf)):
    t_F_L.append(F_L_Pair(tf[i], tl[i]))
del tf
del tl

v_F_L = []
for i in range(len(vf)):
    v_F_L.append(F_L_Pair(vf[i], vl[i]))
del vf
del vl

test_F_L = []
for i in range(len(testf)):
    test_F_L.append(F_L_Pair(testf[i], testl[i]))
del testf
del testl

def v_file_name(k):
    return "digitsOutput" + str(k) + ".csv"

def test_file_name():
    return "digitsOutput.csv"

kv1 = classify(1, v_file_name(1), v_F_L, True)
kv2 = classify(2, v_file_name(2), v_F_L, True)
kv5 = classify(5, v_file_name(5), v_F_L, True)
kv10 = classify(10, v_file_name(10), v_F_L, True)
kv25 = classify(25, v_file_name(25), v_F_L, True)

best_k = 5
test_the_set = classify(best_k, test_file_name(), test_F_L, False)

testFeatures.close()
trainFeatures.close()
trainLabels.close()
valFeatures.close()
valLabels.close()
