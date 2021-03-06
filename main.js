const ML = require('./node_modules/machine_learning/lib/machine_learning')

class Block {
    constructor(index, timestamp, data, result) {
        this.index = index
        this.timestamp = timestamp
        this.data = data
        this.result = result
    }

    calculateHash() {
        ML.KNN(this.data)

        let result = this.result

        let knn = new ML.KNN({
            data: this.data,
            result: this.result
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

class BlockChain {
    constructor() {
        this.chain = []
    }

    createGeneisBlock() {
        return this.chain.push(new Block(0,
            '7-6-2018 Genesis Block', [
                [0, 0]
            ], '0')
        )
    }

    getLatestBlock() {
        return this.chain(this.chain.length - 1)
    }

    addBlock(block) {
        block.result = block.calculateHash()
        return this.chain.push(block)
    }
}

let dimChain = new BlockChain()
dimChain.createGeneisBlock()
dimChain.addBlock(new Block(
    dimChain.chain.length,
    new Date, [
        [1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0],
        [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0],
        [1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0],
        [1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0],
        [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1],
        [0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0],
        [0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0]
    ], [23, 12, 23, 23, 45, 70, 123, 73, 146, 158, 64]
))
dimChain.addBlock(new Block(
    dimChain.chain.length,
    new Date, [
        [1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0],
        [1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0],
        [1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0],
        [1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0]
    ], [23, 12, 23, 23, 75, 70]
))

console.log(dimChain)