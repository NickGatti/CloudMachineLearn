const ML= require('./node_modules/machine_learning/lib/machine_learning')

class Block{
    constructor(index, timestamp, data, previousHash = ''){
        this.index = index
        this.timestamp = timestamp
        this.data = data
        this.previousHash = previousHash
        this.hash = hash
    }

    calculateHash(){
        ML.KNN(this.data)

        let knn = new ML.KNN({
            data: this.data,
            result: result
        })

        let y = knn.predict({
            x: [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
            k: 3,
            weightf: {
                type: 'gaussian',
                sigma: 10.0
            },
            distance: {
                type: 'euclidean'
            }
        });

        return y
    }
}