# Instalar paquetes
$ sudo apt update

$ sudo apt install git

$ sudo apt install npm 

# Clonar repositorio 
$ sudo git clone https://github.com/melissasoriano98/proyectoIA.git

# Dentro de la carpeta del proyecto 
$ npm i

# Crear extras
 Crear la carpeta files dentro del proyecto
 ![image](https://user-images.githubusercontent.com/34258748/126432911-b19e3012-80e0-420e-8116-f9271a1fe753.png)
 
# Correr proyecto 
$ npm start 
![image](https://user-images.githubusercontent.com/34258748/126432526-0601ddbb-6392-48a5-bfdd-fec6bed5b85e.png)

# Descargar e Instalar Postman
https://www.postman.com/downloads/

# Asignar el dato a predecir
En la línea 167 del archivo neural_network.js, se podrá editar el valor a predecir. Como en el siguiente caso se está tomando el primer registro de nuestra base de datos

![image](https://user-images.githubusercontent.com/34258748/126433232-68b05fef-1876-4d05-a23e-fc2cdea42fb1.png)


# Hacer petición 
El proyecto corre en localhost puerto 2000, en Postman

Se crea una nueva petición con el método POST y con la url http://localhost:2000/neural-network/

Se agregan los csv en el BODY como form-data, con el nombre features para el csv features.csv y results para el csv respuesta.csv

Y posteriormente también el rango de aprendizaje con el nombre de learningRate y el número de épocas con el nombre numberOfEpochs

Todo como se muestra en la siguiente imágen
![image](https://user-images.githubusercontent.com/34258748/126431988-8e847838-ae31-4a67-9bcf-c70de020f740.png)
