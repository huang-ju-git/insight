import pickle
# make an example object to pickle
some_obj = {'x':[4,2,1.5,1], 'y':[32,[101],17], 'foo':True, 'spam':False}
with open('mypickle.pickle', 'wb') as f:
    pickle.dump(some_obj, f)