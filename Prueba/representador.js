function iniciar() {

    $.ajax({
        cache: false,
        url: "data.json",
        dataType: "json",
        success: function(data) {
            d3.json("data.json", function(data) {                                                       //Se utiliza la librería D3 para poder leer el archivo JSON que nos proporciona el programa principal.d3.json("data.json", function(data) {                                                       //Se utiliza la librería D3 para poder leer el archivo JSON que nos proporciona el programa principal.

                var matriz = [];
                var sequence = [];

                data.forEach(function(item){
                    item.forEach(function(element) {
                        sequence.push(element.x);
                        sequence.push(element.y);
                    });
                    matriz.push(sequence);
                    sequence = [];
                });

                console.log(matriz);



                var particulas = [];

                for (var i = 0; i < data[0].length; i++) {                                             //Bucle para crear todos los círculos.
                    var circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");     //Creación de los círculos en SVG.
                    circle.setAttribute("cx", matriz[0][0 + 2*i]);                                     //Asignación de atributos de los círculos en el instante inicial.
                    circle.setAttribute("cy", 50 - matriz[0][1 + 2*i]);
                    circle.setAttribute("r", 0.6);
                    circle.setAttribute("fill", '#'+Math.round(0xffffff * Math.random()).toString(16));
                    document.getElementById('representador').append(circle);                            //Añadir puntos al SVG.
                    particulas.push(circle);                                                            //Añadir los círculos a una matriz para poder trabajar con ellos en la función movimiento.
                }
                console.log(particulas);



                var i = 1;

                d3.select("#boton1 button").on("click", function movimiento() {                         //Función para el movimiento de las partículas.
                    setInterval( function () {                                                          //Se establece un intervalo de 0,8s entre cada representación.
                        if (i < matriz.length) {
                            for (j = 0; j < particulas.length; j++) {
                                particulas[j].setAttribute("cx", matriz[i][0 + 2*j]);                  //Se modifican las posiciones de los centros de los círculos
                                particulas[j].setAttribute("cy", 50 - matriz[i][1 + 2*j]);                  //para simular el movimiento.
                            }
                        }
                        i++;
                    },50);
                })
            });
        }
    });

}
