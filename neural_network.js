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

            const numTest = Math.round(numExamples * dataSplit)
            const numTrain = numExamples - numTest
            const xDims = shuffledData[0].length
            const xs = tf.tensor2d(shuffledData, [numExamples, xDims], 'int32')

            for(let item of shuffledTargets){ shuffledTargets[shuffledTargets.indexOf(item)] = parseInt(item)}
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
                for(let i = 0; i < entitiesNum; i++){
                    dataByEntitie.push([])
                    targetsByEntitie.push([])
                }
                for(const item of compound){
                    const target = item[item.length -1]
                    const data = item.slice(0, item.length -1)
                    dataByEntitie[entities.indexOf(target)].push(data)
                    targetsByEntitie[entities.indexOf(target)].push(target)
                }

                const xTrains = []
                const yTrains = []
                const xTests = []
                const yTests = []

                for(let i = 0; i < entitiesNum; i++){
                    const [xTrain, yTrain, xTest, yTest] = convertToTensors(dataByEntitie[i], targetsByEntitie[i], dataSplit)
                    xTrains.push(xTrain)
                    yTrains.push(yTrain)
                    xTests.push(xTest)
                    yTests.push(yTest)
                }

                const concatAxis = 0
                const test1 = xTrains
                const test2 = tf.concat(xTrains, concatAxis)
                //console.log(test1)
                //console.log(test2)
                return [
                    tf.concat(xTrains, concatAxis), 
                    tf.concat(yTrains, concatAxis),
                    tf.concat(xTests, concatAxis),
                    tf.concat(yTests, concatAxis)
                ]
            }) 
        }
        //getFeatureData(.2)


        async function trainModel(xTrain, yTrain, xTest, yTest){
            const model = tf.sequential()
            const learningRate = req.body.learningRate
            const numberOfEpochs = req.body.numberOfEpochs
            const optimizer = tf.train.adam(learningRate)

            //console.log(xTrain.shape[1])

            model.add(tf.layers.dense(
                { units:10, activation: 'sigmoid', inputShape: [xTrain.shape[1]]}
            ))
            model.add(tf.layers.dense(
                { units:2, activation: 'softmax'}
            ))
            model.compile(
                { optimizer: optimizer, loss: 'categoricalCrossentropy', metrics: ['accuracy']}
            )

            //console.log(xTrain)
            const history = await model.fit(xTrain, yTrain, 
                { epochs: numberOfEpochs, validationData: [xTest, yTest],
                    callbacks: {
                        onEpochEnd: async (epoch, logs) => {
                            console.log(`Epoch: ${epoch} Logs: ${logs.loss}`)
                            await tf.nextFrame()
                        }
                    }
                }
            )
            return model
        }
        async function doFeature(){
            const [xTrain, yTrain, xTest, yTest] = getFeatureData(.2)
            model = await trainModel(xTrain, yTrain, xTest, yTest)

            const input = tf.tensor2d([5,1,1,1,2,1,3,1,1], [1, 9])
            const prediction = model.predict(input)
            console.log(prediction)

            const predictionWithArgMax = model.predict(input).argMax(-1).dataSync()
            console.log(predictionWithArgMax)

            const xData = xTest.dataSync()
            const yTrue = yTest.argMax(-1).dataSync()

            const predictions = await model.predict(xTest)
            const yPred = predictions.argMax(-1).dataSync()
            
            let correct = 0
            let wrong = 0
            for (let i = 0; i < yTrue.length; i++) {
                //console.log(`True: ${yTrue[i]}, Pred: ${yPred[i]}`)
                if(yTrue[i] == yPred[i]){
                    correct++
                }else{
                    wrong++
                }   
            }
            console.log(`Prediction error rate: ${wrong/yTrue.length}`)
            let Prediction_Error_Rate = wrong/yTrue.length

            let data = entities[predictionWithArgMax]

            return {
                Corrects: correct,
                Wrongs: wrong,
                Prediction_Error_Rate: Prediction_Error_Rate,
                Result: data
            }
        }

        let result = await doFeature()
        return res.status(200).json({result})
    }catch(error){
        console.log(error)
        next(error)
    }
})

module.exports = route