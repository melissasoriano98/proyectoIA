const express = require('express')
const route = express.Router()
const tf = require('@tensorflow/tfjs')
const fs = require("fs")
const multer = require("multer")
const storage = multer.diskStorage({
    destination: function(req, file, cb){
        cb(null, 'files')
    },
    filename: function(req, file, cb){
        let extension = (file.mimetype).split('/')
        cb(null, `${file.originalname}.${extension[1]}`)
    }
})

const upload = multer({
    storage: storage
})

route.post('/', upload.fields([{name: 'features'},{name: 'results'}]),  async(req, res, next) =>{
    try{
        let compound = []

        let featureRows = fs.readFileSync(req.files.features[0].path)
            .toString() // convert Buffer to string
            .split('\n') // split string to lines
            .map(e => e.trim()) // remove white spaces for each line
            .map(e => e.split(',').map(e => e.trim())); // split each line to array
        featureRows.pop()
        compound = featureRows

        let resultRows = fs.readFileSync(req.files.results[0].path)
            .toString() // convert Buffer to string
            .split('\n') // split string to lines
            .map(e => e.trim()) // remove white spaces for each line
            .map(e => e.split(',').map(e => e.trim())); // split each line to array
        resultRows.pop()

        for(let item of compound){
            item.push(resultRows[compound.indexOf(item)][0])
        }

        const entities = resultRows.reduce((acc,item)=>{
            let a = acc.toString()
            let b = item.toString()
            if(!a.includes(b)){
                acc.push(b);
            }
            return acc;
        })

        const entitiesNum = entities.length

        console.log(entities)

        function convertToTensors(data, targets, dataSplit){
            const numExamples = data.length
            if(numExamples !== data.length) throw new Error('Data and split have different numbers of examples')

            // Randomly shuffle `data` and `targets`.
            const indices = [];
            for (let i = 0; i < numExamples; ++i) {
                indices.push(i);
            }
            tf.util.shuffle(indices);

            const shuffledData = [];
            const shuffledTargets = [];
            for (let i = 0; i < numExamples; ++i) {
                shuffledData.push(data[indices[i]]);
                shuffledTargets.push(targets[indices[i]]);
            }

            console.log(shuffledData)

            const numTest = Math.round(numExamples * dataSplit)
            const numTrain = numExamples - numTest
            const xDims = shuffledData[0].length

            const xs = tf.tensor2d(shuffledData, [numExamples, xDims])
            const ys = tf.oneHot(tf.tensor1d(shuffledTargets).toInt(), entitiesNum)

            const xTrain = xs.slice([0,0], [numTrain, xDims])
            const yTrain = ys.slice([0,0], [numTrain, entitiesNum])
            const xTest = xs.slice([numTrain, 0], [numTest, xDims])
            const yTest = ys.slice([0,0], [numTest, entitiesNum])

            return [xTrain, yTrain, xTest, yTest]
        }

        function getFeatureData(dataSplit){
            return tf.tidy(()=> {
                const dataByEntitie = []
                const targetsByEntitie = []
                for(let i = 0; i < 10; i++){
                    dataByEntitie.push([])
                    targetsByEntitie.push([])
                }
                for(const item of compound){
                    const target = item[item.length -1]
                    const data = item.slice(0, item.length -1)
                    dataByEntitie[target].push(data)
                    targetsByEntitie[target].push(target)
                }

                const xTrains = []
                const yTrains = []
                const xTests = []
                const yTests = []

                for(let i = 0; i < 10; i++){
                    const [xTrain, yTrain, xTest, yTest] = convertToTensors(dataByEntitie[i], targetsByEntitie[i], dataSplit)
                    xTrains.push(xTrain)
                    yTrains.push(yTrain)
                    xTests.push(xTest)
                    yTests.push(yTest)
                }

                const concatAxis = 0
                const test1 = xTrains
                const test2 = tf.concat(xTrains, concatAxis)
                console.log(test1)
                console.log(test2)
                return [
                    tf.concat(xTrains, concatAxis),
                    tf.concat(yTrains, concatAxis),
                    tf.concat(xTests, concatAxis),
                    tf.concat(yTests, concatAxis)
                ]
            }) 
        }
        getFeatureData(.2)
        return res.status(200).json({compound})
    }catch(error){
        console.log(error)
        next(error)
    }
})

module.exports = route