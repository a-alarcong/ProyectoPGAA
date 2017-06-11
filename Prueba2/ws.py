import tornado.httpserver
import tornado.websocket
import tornado.ioloop
import tornado.web
import json
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm, det
import sys




###AQUI ESTAN LOS MODULOS DE NUESTRO PROGRAMA####




###AQUI EMPIEZA LA INTERACCION JS-PYTHON###

class IndexHandler(tornado.web.RequestHandler):
    @tornado.web.asynchronous
    def get(self):
        #self.write("This is your response")
        #self.finish()
        self.render("index.html")

class WSHandler(tornado.websocket.WebSocketHandler):
    connections = set()

    def open(self):
        self.connections.add(self)
        print('New connection was opened')
        self.write_message("Conn!")

    def on_message(self, message):

        json_string = u'%s' %(message,)

        mensajedeJS = json.loads(json_string)

        msg_particulas = mensajedeJS['particulas']
        msg_tiempo_foco = mensajedeJS['tiempo_foco']



        n = int(msg_particulas)
        t_in = int(msg_tiempo_foco)


###AQUI EMPIEZA EL PROGRAMA PRINCIPAL###

        def vec_aleat(k):
            if k == 1:

                vec = [np.random.randint(0, 50), np.random.randint(0, 50)]
                # print(vec)

            elif k == 2:

                vec = [np.random.randint(-10, 10), np.random.randint(-10, 10)]
                # print(vec)

            return vec

        class cc():
            def __init__(self):

                self.top = 50
                self.bottom = 0
                self.left = 0
                self.right = 50
                self.e = 1  # coef restitución
                self.sale = False
                self.huida = 0

            def paredes(self, g_part, i, t, t_in, matrizposicionx, matrizposiciony, stop):

                if t >= t_in:

                    if g_part[i].posicion[0] < self.left and matrizposiciony[t, i] != self.top and matrizposiciony[
                        t, i] != self.bottom:

                        for h in range(t, stop):
                            matrizposicionx[h, i] = self.left
                            matrizposiciony[h, i] = matrizposiciony[h - 1, i]

                        self.huida = 1

                    elif g_part[i].posicion[0] > self.right and matrizposiciony[t, i] != self.top and matrizposiciony[
                        t, i] != self.bottom:

                        for h in range(t, stop):
                            matrizposicionx[h, i] = self.right
                            matrizposiciony[h, i] = matrizposiciony[h - 1, i]
                        self.huida = 1

                    if g_part[i].posicion[1] > self.top and matrizposicionx[t, i] != self.right and matrizposicionx[
                        t, i] != self.left:

                        for h in range(t, stop):
                            matrizposiciony[h, i] = self.top
                            matrizposicionx[h, i] = matrizposicionx[h - 1, i]
                        self.huida = 1

                    elif g_part[i].posicion[1] < self.bottom and matrizposicionx[t, i] != self.right and \
                                    matrizposicionx[
                                        t, i] != self.left:

                        for h in range(t, stop):
                            matrizposiciony[h, i] = self.bottom
                            matrizposicionx[h, i] = matrizposicionx[h - 1, i]
                        self.huida = 1

                    if self.huida == 1:

                        print('La particula ' + str(i) + ' ha salido')

                    else:

                        matrizposicionx[t, i] = g_part[i].posicion[0]
                        matrizposiciony[t, i] = g_part[i].posicion[1]

                    return self.huida, matrizposicionx, matrizposiciony


                else:

                    if g_part[i].posicion[0] < self.left or g_part[i].posicion[0] > self.right:

                        g_part[i].velocidad[0] = -self.e * g_part[i].velocidad[0]

                        if g_part[i].posicion[0] > self.right:

                            g_part[i].posicion[0] = self.right - (g_part[i].posicion[0] - self.right)

                        elif g_part[i].posicion[0] < self.left:

                            g_part[i].posicion[0] = self.left - (g_part[i].posicion[0] - self.left)

                        self.sale = True

                        print("Se saleeee")

                    if g_part[i].posicion[1] < self.bottom or g_part[i].posicion[1] > self.top:

                        g_part[i].velocidad[1] = -self.e * g_part[i].velocidad[1]

                        if g_part[i].posicion[1] > self.top:

                            g_part[i].posicion[1] = self.top - (g_part[i].posicion[1] - self.top)

                        elif g_part[i].posicion[1] < self.bottom:

                            g_part[i].posicion[1] = self.bottom - (g_part[i].posicion[1] - self.bottom)

                        self.sale = True

                        print("Se saleeee")

                    matrizposicionx[t, i] = np.array(g_part[i].posicion[0])
                    matrizposiciony[t, i] = np.array(g_part[i].posicion[1])

                    return self.sale, matrizposicionx, matrizposiciony

        class metodos():
            def __init__(self, n, g_part, vel, dt, alpha, t, t_in):

                self.n = n
                self.theta_m = 0
                # self.thetas = np.zeros(n)
                self.s_x = 0
                self.s_y = 0
                self.vel = vel
                self.v_act = 0

            def thetamed(self):

                for i in range(n):

                    s_x = g_part[i].velocidad[0]
                    s_y = g_part[i].velocidad[1]

                    '''''
                    for j in range(n):

                        if 0 < norm(np.array(g_part[i].posicion) - np.array(g_part[j].posicion))< r:

                            s_x = s_x + g_part[j].velocidad[0]
                            s_y = s_y + g_part[j].velocidad[1]

                            print('j',j)
                    '''''
                    if t >= t_in:
                        s_x = s_x + (g_part[i].posicion[0] - pos_foco[0])
                        s_y = s_y + (g_part[i].posicion[1] - pos_foco[1])

                    # print("sx",s_x,"v",g_part[i].velocidad)

                    if abs(s_x) < 0.0001 and s_y > 0:

                        theta_m = np.pi / 2

                    elif abs(s_x) < 0.0001 and s_y < 0:

                        theta_m = 3 * np.pi / 2

                    elif abs(s_y) < 0.0001 and s_x < 0:

                        theta_m = np.pi

                    elif abs(s_y) < 0.0001 and s_x > 0:

                        theta_m = 0

                    else:

                        theta_m = np.arctan(s_y / s_x)

                        if theta_m < 0 and s_x < 0:

                            theta_m = np.pi + theta_m

                        elif theta_m > 0 and s_x < 0:

                            theta_m = np.pi + theta_m

                    print("theta_m", np.degrees(theta_m))

                    g_part[i].angulo = theta_m

                    # return g_part.angulo

            def actualiza(self, matrizposicionx, matrizposiciony):

                # actualiza velocidad y vector de posicion

                acum = 0

                for i in range(n):
                    g_part[i].angulo = g_part[i].angulo + np.random.uniform(-0.2, 0.2)  # actualiza tetha con ruido

                    print("ang con ruido", np.degrees(g_part[i].angulo))
                    g_part[i].velocidad = alpha * g_part[i].velocidad + \
                                          (1 - alpha) * self.vel * np.array(
                                              [np.cos(g_part[i].angulo), np.sin(g_part[i].angulo)])
                    g_part[i].posicion = g_part[i].posicion + dt * g_part[i].velocidad

                    Cond = cc()
                    Cond.paredes(g_part, i, t, t_in, matrizposicionx, matrizposiciony, stop)

                    print('HUIDA VALE: ' + str(Cond.huida))
                    # print("¿Se sale? "+ str(Cond.sale))

                    if Cond.huida == 1:

                        acum = acum + 1

                        if acum == n:

                            print('TODAS LAS PARTICULAS SALIERON')

                            print(matrizposicionx)

                            print(matrizposiciony)




                            for v in range(stop):

                                for b in range(n):
                                    matrizposicionx[v, b] = np.round(matrizposicionx[v, b], 2)
                                    matrizposiciony[v, b] = np.round(matrizposiciony[v, b], 2)




                            file = open('data.json', 'w')
                            file.write('[\n')
                            for f in range(stop):
                                line = '[{"x": ' + str(pos_foco[0]) + ', "y": ' + str(pos_foco[1]) + "},"
                                for g in range(n):
                                    line = line + '{"x":' + str(matrizposicionx[f, g]) + ', "y": ' + str(
                                        matrizposiciony[f, g]) + "},"

                                line = line[:-1]
                                line = line + "],"
                                if f == (stop - 1):
                                    line = line[:-1]

                                file.write(line + '\n')

                            file.write(']\n')
                            file.close()

                            #plt.show()

                            #sys.exit()

                # matrizposicionx = np.array[range(n) for i in range(stop)



                return matrizposicionx, matrizposiciony

        class visual():
            def __init__(self, g_part, indice, pos_foco):
                self.i = indice
                self.colores = ['or', 'ob', 'om', 'oy', 'og', '*r', '*b', '*m', '*y', '*g']

            def dibuja(self):
                plt.plot(g_part[i].posicion[0], g_part[i].posicion[1], self.colores[i])
                plt.plot(pos_foco[0], pos_foco[1], '*y')

            def datos_vuelta(self, t):
                if i == 0:
                    print("------VUELTA " + str(t) + "------")

                print("Velocidad " + str(i) + " : ", g_part[i].velocidad)
                print("Posición " + str(i) + " : ", g_part[i].posicion)
                print("Ángulo " + str(i) + " : ", np.degrees(g_part[i].angulo))

        class particula():
            def __init__(self, v_o, r, pos, v):
                self.radio = r
                self.v_o = v_o

                # self.v = v
                # self.pos = pos

            def posicion(self):

                self.posicion = pos
                print("jeje")
                return self.posicion

            def velocidad(self):

                self.velocidad = v / norm(v)
                print("jiji")
                return self.velocidad

            def angulo(self):

                if v[0] == 0:

                    self.angulo = np.pi / 2

                else:

                    self.angulo = np.arctan(v[1] / v[0])

                return self.angulo



            ###AQUI EMPIEZA EL PROGRAMA PRINCIPAL####


        v_o = 1
        # t = 0
        # n = 5
        r = 3
        dt = 1
        stop = 50
        alpha = 0.3  # Inercia a mantener la dirección del instante anterior

        pos_foco = vec_aleat(1)
        p = np.zeros(n)
        pos = np.zeros(n)
        v = np.zeros(n)
        g_part = []
        print(g_part)

        matrizposicionx = np.empty((stop, n))
        matrizposiciony = np.empty((stop, n))
            # matrizposiciony = [range(n) for i in range(stop)]





        for i in range(0, n):
                pos = vec_aleat(1)
                v = vec_aleat(2)
                p = particula(v_o, r, pos, v)
                print(pos)
                p.posicion()  # si no se pone el argumento de dentro no va
                p.velocidad()
                p.angulo()
                print("La posicion de la particula es: " + np.str(p.posicion))
                print("La velocidad de la particula es: " + np.str(p.velocidad))

                g_part.append(p)

                #######visual(g_part, i, pos_foco).dibuja()

            # print(g_part[0].posicion)
            # print(g_part[1].velocidad)



        #plt.xlim(0, 50)
        #plt.ylim(0, 50)

        #fig = plt.figure()

        plt.xlim(0, 50)
        plt.ylim(0, 50)


        for t in range(stop):

                Play = metodos(n, g_part, v_o, dt, alpha, t, t_in)
                Play.thetamed()
                # print("Thetas med ", g_part.angulo)
                Play.actualiza(matrizposicionx, matrizposiciony)
                # print("x", g_part[0].posicion, g_part[1].posicion, g_part[2].posicion, g_part[3].posicion)

                # print("t",t)
                # print("tethas",np.degrees(g_part.angulo))

                for i in range(n):
                    visual(g_part, i, pos_foco).dibuja()
                    visual(g_part, i, pos_foco).datos_vuelta(t)
                    # print("tethas", np.degrees(g_part[i].angulo))

        print(matrizposicionx)

        print(matrizposiciony)




        for v in range(stop):

            for b in range(n):
                matrizposicionx[v, b] = np.round(matrizposicionx[v, b], 2)
                matrizposiciony[v, b] = np.round(matrizposiciony[v, b], 2)




        file = open('data.json', 'w')
        file.write('[\n')
        for f in range(stop):
            line = '[{"x": ' + str(pos_foco[0]) + ', "y": ' + str(pos_foco[1]) + "},"
            for g in range(n):
                line = line + '{"x":' + str(matrizposicionx[f, g]) + ', "y": ' + str(matrizposiciony[f, g]) + "},"

            line = line[:-1]
            line = line + "],"
            if f == (stop - 1):
                line = line[:-1]

            file.write(line + '\n')

        file.write(']\n')
        file.close()

        #plt.show()


        #print(self.connections)
        for elem in self.connections:
            elem.write_message({"msg": message + ' a todos'}, binary=False)

        # [con.write_message('Hi!') for con in self.connections]
        self.write_message({"msg": message + ' al que lo manda'})


    def on_close(self):
        print('connection closed\n')



application = tornado.web.Application([(r'/', IndexHandler),(r'/ws', WSHandler),
(r'/(.*)', tornado.web.StaticFileHandler, {'path': r'C:/python3'})])

if __name__ == "__main__":
    http_server = tornado.httpserver.HTTPServer(application)
    http_server.listen(8888)
    tornado.ioloop.IOLoop.instance().start()