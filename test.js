const ML= require('./node_modules/machine_learning/lib/machine_learning')

class Block{
    constructor(index, timestamp, data, result){
        this.index = index
        this.timestamp = timestamp
        this.data = data
        this.result = result
    }

    calculateHash(){
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

let dimChain = new Block(
    0,
    '7-6-2018', 
    [
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
    ],
    [23, 12, 23, 23, 45, 70, 123, 73, 146, 158, 64]
)
console.log(dimChain)
console.log(JSON.stringify(dimChain.calculateHash()))

/*
class BlockChain{
    constructor(){
        this.chain = []
    }

    createGeneisBlock(){
        return new Block(0, '7-6-2018', [ [0, 1], [0, 1] ], '0')
    }

    getLatestBlock(){
        return this.chain(this.chain.length - 1)
    }

    addBlock(){
        newBlock.previousHash = this.getLatestBlock().hash
        newBlock.hash = newBlock.calculateHash()
        this.chain.push(newBlock)
    }
}

let dimChain = new BlockChain()
dimChain.addBlock(new Block( 
    1,
    '7-6-2018',
    { data: [
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
    ]}
 ))

 console.log(JSON.stringify(dimChain, null, [
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
 ]))
 
 console.log(dimChain)
 console.log(JSON.stringify(dimChain.calculateHash()))
*/