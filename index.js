const express = require('express')
const app = express()
const port = 2000
const neuralNetwork = require('./neural_network') 

app.use(express.json())
app.use('/neural-network', neuralNetwork)

app.listen(port, function () {
    console.log(`'Proyecto-IA' server listening on port ${port}`)
})