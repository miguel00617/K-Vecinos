Predicción con K-Vecinos mas cercanos
Proyecto de algoritmo de clasificación de Machine Learning

Este proyecto de ciencia de datos, busca agilizar la clasificación de un deudor de alguna entidad bancaria, comercial, etc; donde dependiendo su edad y el monto, se define si este deudor será responsable y pagará o no, por medio de una gráfica donde se explica mejor su posicionamiento.

Primero importamos las librerías, las herramientas, luego, importamos el data set, con los datos para entrenar el modelo(se encuentra en Data, como "creditos.csv").

Realizamos luego una clasificacion de los clientes, en dos, buenos y malos, por un tema de identificación, tenieendo en cuenta si pagaron o no su deuda. Por siguiente una visualización de esta clasificación, con un gráfico de scatterplot para ver mejor la relación y distribución de los datos, separando a ambas clases de clientes, por colores, para ver donde se agrupan y concentran.

Por siguiente creamos el modelo, lo entrenamos, y vemos cual es su rango de error de predicción. Le damos otro dato para predecir y por último producimos nuevos solicitantes, y el modleo predictivo los clasfica y agrupa en un gráfico, para observar la tendencia y el aglomeramiento de datos.

#Tencnologías: (Python: Numpy, Pandas, Matplotlib, Scikit-learn y sklearn.neighbors) #Fuente de Datos: Datasets creado en ChatGPT(dataset: creditos.csv).
