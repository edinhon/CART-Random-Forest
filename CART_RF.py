import numpy as np

class CART: 
    def __init__(self): 
        self.tree = None

    def gini_index(self, data, fidx):
        sorted_data = np.array(sorted(data, key=lambda x: x[fidx]))
        n = len(sorted_data)
        if n == 1: 
            return 0, 0

        nPos, nNeg = (sorted_data[:, -1]==1.).sum(), (sorted_data[:, -1]==-1.).sum()
        curPos, curNeg = (sorted_data[0, -1]==1.).sum(), (sorted_data[0, -1]==-1.).sum()
        ginis, medians = [], [(sorted_data[i, fidx]+sorted_data[i+1, fidx])/2 for i in range(0, n-1)]
        for i in range(0, n-1): 
            g1 = 1. - (curPos/(i+1))**2 - (curNeg/(i+1))**2 if i+1 != 0 else 0.
            g2 = 1. - ((nPos-curPos)/(n-i-1))**2 - ((nNeg-curNeg)/(n-i-1))**2 if (n-i-1) != 0 else 0.
            ginis.append((i+1)*g1/n + (n-i-1)*g2/n)
            if sorted_data[i+1, -1] == 1.: 
                curPos += 1
            else: 
                curNeg += 1

        return np.min(ginis), medians[np.argmin(ginis)]
    
    def fit(self, X, Y):
        data = np.concatenate((X, np.expand_dims(Y, axis=0).T), axis=1)
        self.tree = self._apply(data)

    def _apply(self, data):
        n, k = data.shape[0], data.shape[1]-1
        if ((data[:, -1]==1.).sum()) == n or ((data[:, -1]==-1.).sum() == n): 
            return {'feature': -1, 'class': data[0, -1], 'subs': None}

        ginis, thetas = [], []
        for i in range(k):
            gini, seg = self.gini_index(data, i)
            ginis.append(gini)
            thetas.append(seg)

        minK = np.argmin(ginis)
        minT = thetas[minK]
        sub1, sub2 = data[data[:, minK] >= minT], data[data[:, minK] < minT]
        return {'feature': minK, 'threshold': minT, 'subs': [self._apply(sub1), self._apply(sub2)]}

    def predict(self, data): 
        rst = []
        for i in range(len(data)): 
            rst.append(self._predict(data[i]))
        return rst

    def _predict(self, x): 
        tree = self.tree
        while tree['subs'] != None: 
            if x[tree['feature']] >= tree['threshold']: 
                tree = tree['subs'][0]
            else: 
                tree = tree['subs'][1]
        return tree['class']

class RandomForest:
    
    def __init__(self, n_estimators=100, max_samples=0.5):
        self.n_estimators = n_estimators
        self.estimators = [CART() for _ in range(n_estimators)]
        self.frac = max_samples
        self.samples = []

    def fit(self, X, Y, quiet=False): 
        n = len(X)
        data = np.concatenate((X, np.expand_dims(Y, axis=0).T), axis=1)
        for i in range(self.n_estimators): 
            if quiet == False: 
                print(f'fitting {i} tree...', end='\r')
            sample = np.random.choice(n, size=int(self.frac*n), replace=True)
            subdata = np.array([data[x] for x in sample])
            self.estimators[i].fit(subdata[:, :-1], subdata[:, -1])
            self.samples.append(sample)
        print('')

    def predict(self, X):
        preds = []
        for i in range(self.n_estimators): 
            preds.append(self.estimators[i].predict(X))

        rst = np.sum(preds, axis=0)
        return np.sign(rst)

    def mean_score(self, X, Y): 
        scores = []
        for i in range(self.n_estimators): 
            pred = self.estimators[i].predict(X)
            scores.append(accuracy(pred, Y))
        return np.mean(scores)

    def oob_score(self, X, Y): 
        score = 0
        for i in range(len(X)): 
            pred_sum = 0
            for j in range(self.n_estimators): 
                if i not in self.samples[j]: 
                    pred = self.estimators[j].predict([X[i]])
                    pred_sum += pred[0]
            if np.sign(pred_sum) == Y[i]: 
                score += 1
        return score/len(X)

def accuracy(pred, truth): 
    pred = np.array(pred)
    truth = np.array(truth)
    return (pred == truth).sum()/len(pred)

def main(): 
    train_data = np.loadtxt('./hw6_train.dat')
    test_data = np.loadtxt('./hw6_test.dat')
    n, k = train_data.shape[0], train_data.shape[1]-1
    
    trainX, trainY = train_data[:, :-1], train_data[:, -1]
    testX, testY = test_data[:, :-1], test_data[:, -1]
    
    # P14
    clf = CART()
    clf.fit(trainX, trainY)
    pred = clf.predict(testX)
    print(f'P14: Out Accuracy = {accuracy(pred, testY)}')

    # P15 P16 P17 P18
    clf = RandomForest(n_estimators=2000, max_samples=0.5)
    clf.fit(trainX, trainY)
    print(f'P15: Mean Score = {clf.mean_score(testX, testY)}')
    pred_in = clf.predict(trainX)
    print(f'P16: In Accuracy = {accuracy(pred_in, trainY)}')
    pred_out = clf.predict(testX)
    print(f'P17: Out Accuracy = {accuracy(pred_out, testY)}')
    print(f'P18: OOB Score = {clf.oob_score(trainX, trainY)}')

if __name__ == '__main__': 
    main()
