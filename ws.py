import tornado.httpserver
import tornado.websocket
import tornado.ioloop
import tornado.web
import json


import matplotlib.pyplot as plt

from numpy.linalg import norm, det

import math

import matplotlib.animation as manimation

import openpyxl
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
        msg_tipo = mensajedeJS['tipo']

        n = int(msg_particulas)
        t_in = int(msg_tiempo_foco)

        if msg_tipo == 'evacuacion':

            import numpy as np

            import matplotlib.pyplot as plt

            from numpy.linalg import norm, det

            import math

            import matplotlib.animation as manimation

            import openpyxl

            # FFMpegWriter = manimation.writers['ffmpeg']

            # writer = FFMpegWriter(fps = 15)





            def vec_aleat(k):

                if k == 1:

                    vec = [np.random.randint(5, 45), np.random.randint(40, 45)]

                    # print(vec)



                elif k == 2:

                    vec = [np.random.randint(-10, 10), np.random.randint(-10, 10)]

                    # print(vec)

                return vec



            def dist_r_P(A_r, B_r, C_r, Punto):

                raiz = math.sqrt(A_r ** 2 + B_r ** 2)

                dist = (abs(A_r * Punto[0] + B_r * Punto[1] + C_r)) / raiz

                return dist

            ### DEFINE RECTA DEL TIPO Ax + By + C = 0 ###



            class pared():

                def __init__(self, vector, punto, comienzo_hueco, fin_hueco):
                    self.u_r = vector

                    self.p_r = punto

                    self.A = 0

                    self.B = 0

                    self.C = 0

                    self.h_c = comienzo_hueco

                    self.h_f = fin_hueco

                def r(self):
                    self.A = self.u_r[1]

                    self.B = -self.u_r[0]

                    self.C = self.u_r[0] * self.p_r[1] - self.u_r[1] * self.p_r[0]

                    self.l_h = norm(self.h_c - self.h_f)  # longitud del hueco

                    return

            class cc():

                def __init__(self):

                    self.top = 50

                    self.bottom = 0

                    self.left = 0

                    self.right = 50

                    self.e = 1  # coef restitución

                    self.sale = False

                    self.d1 = 0

                    self.huec = 1

                def f_particulas(self, i, A, B, k, kappa, r_p):

                    f_p = np.array([0, 0])

                    f_p_final = np.array([0, 0])

                    for j in range(n):

                        if j == i:

                            continue



                        else:

                            d = norm(np.array(g_part[i].posicion) - np.array(g_part[j].posicion))

                            if d > (2 * r_p):

                                g = 0



                            else:

                                g = 2 * r_p - d

                            print('VALORES particulas')

                            print(A, r_p, d, B, k, g)

                            f_p_n = A * math.exp((2 * r_p - d) / B) + k * g

                            dir_n = (np.array(g_part[i].posicion) - np.array(g_part[j].posicion)) / d

                            # Sin fricción ni compresibilidad



                            "Ver que la g no sale 0"

                            # f_p_n = A*math.exp((2*r_p - d)/B)





                            # f_p_t = kappa * g * (g_part[j].velocidad - g_part[i].velocidad)



                            # dir_t = np.array([-dir_n[1], dir_n[0]])





                            ###NOTA: AQUI FALTA LA DIRECCION TANGENCIAL PERO NO SABIA MUY BIEN QUE DIRECCION TENIA Y NO LA HE PUESTO

                            ##DE TODAS FORMAS SI NO SE CHOCAN NO AFECTA ASI QUE NO CAMBIA NADA LOS CALCULOS MIENTRAS NO SE CHOQUEN

                            f_p = f_p_n * dir_n  # + f_p_t * dir_t

                            f_p_final = (f_p_final + f_p)

                    f_p_final = f_p_final / masa

                    print(f_p_final)

                    return f_p_final

                def f_paredes(self, i, A, B, k, kappa, r_p):

                    f_left = [0, 0]

                    f_right = [0, 0]

                    f_top = [0, 0]

                    f_bottom = [0, 0]

                    f_wall = [0, 0]

                    # AQUI CON LAS PAREDES HAY QUE TENER CUIDADO CON LOS SIGNOS





                    d_left = g_part[i].posicion[0] - self.left

                    d_right = self.right - g_part[i].posicion[0]

                    d_top = self.top - g_part[i].posicion[1]

                    d_bottom = g_part[i].posicion[1] - self.bottom

                    self.d1 = dist_r_P(Tramo1.A, Tramo1.B, Tramo1.C, g_part[i].posicion)

                    n_tr = np.array([-Tramo1.B, Tramo1.A]) / np.sqrt(Tramo1.A ** 2 + Tramo1.B ** 2)

                    # vector normal de la recta



                    vec_aux = Tramo1.p_r - np.array(g_part[i].posicion)

                    # vector auxiliar que apunta a la partícula desde el punto de la recta usado



                    if np.arccos((n_tr[0] * vec_aux[0] + n_tr[1] * vec_aux[1]) / \
 \
                                         (norm(n_tr) * norm(vec_aux))) > np.pi / 2:
                        n_tr = -n_tr

                        # se cambia de sentido si no está en la parte a la que apunta el vector normal

                    if norm((-n_tr * self.d1 + g_part[i].posicion) - Tramo1.h_c) < Tramo1.l_h and norm(
                                    (-n_tr * self.d1 + g_part[i].posicion) - Tramo1.h_f) < Tramo1.l_h:

                        # se tiene en cuenta la fuerza de las esquinas

                        f_tramo = (A * math.exp((r_w - norm(Tramo1.h_c - np.array(g_part[i].posicion))) / B)) * (
                        (np.array(g_part[i].posicion) - Tramo1.h_c))

                        f_tramo = f_tramo + (A * math.exp(
                            (r_w - norm(Tramo1.h_f - np.array(g_part[i].posicion))) / B)) * (
                                            (np.array(g_part[i].posicion) - Tramo1.h_f))





                    else:

                        # fuerza general de la pared

                        f_tramo = (A * math.exp((r_w - self.d1) / B)) * n_tr

                    print("FFFF", norm(f_tramo))

                    ###FUERZA CON LA PARED DE LA IZQUIERDA####





                    if d_left > r_p:

                        g = 0



                    else:

                        g = r_p - d_left

                    print('VALORES')

                    print(A, r_p, d_left, B, k, g)

                    f_left_n = A * math.exp((r_w - d_left) / B) + k * g

                    f_left_t = A * math.exp((r_w - d_left) / B) * math.copysign(1, g_part[i].velocidad[1]) - kappa * g * \
                                                                                                             g_part[
                                                                                                                 i].velocidad[
                                                                                                                 1]

                    f_left = np.array([f_left_n, f_left_t])

                    ###FUERZA CON LA PARED DE LA DERECHA###



                    if d_right > r_p:

                        g = 0



                    else:

                        g = r_p - d_right

                    f_right_n = A * math.exp((r_w - d_right) / B) + k * g

                    f_right_t = A * math.exp((r_w - d_right) / B) * math.copysign(1,
                                                                                  g_part[i].velocidad[1]) - kappa * g * \
                                                                                                            g_part[
                                                                                                                i].velocidad[
                                                                                                                1]

                    f_right = np.array([- f_right_n, f_right_t])

                    ###FUERZA CON LA PARED DE ARRIBA####



                    if d_top > r_p:

                        g = 0



                    else:

                        g = r_p - d_top

                    f_top_n = A * math.exp((r_w - d_top) / B) + k * g

                    f_top_t = A * math.exp((r_w - d_top) / B) * math.copysign(1, g_part[i].velocidad[0]) - kappa * g * \
                                                                                                           g_part[
                                                                                                               i].velocidad[
                                                                                                               0]

                    f_top = np.array([f_top_t, -f_top_n])

                    ###FUERZA CON LA PARED DE ABAJO###



                    if d_bottom > r_p:

                        g = 0



                    else:

                        g = r_p - d_bottom

                    f_bottom_n = A * math.exp((r_w - d_bottom) / B) + k * g

                    f_bottom_t = A * math.exp((r_w - d_bottom) / B) * math.copysign(1, g_part[i].velocidad[
                        0]) - kappa * g * g_part[i].velocidad[0]

                    f_bottom = np.array([f_bottom_t, f_bottom_n])

                    f_wall = (f_bottom + f_top + f_right + f_left + f_tramo) / masa

                    # hoja.append([t, i, norm(f_tramo), norm(f_bottom), norm(f_top) , norm(f_right), norm(f_left), norm(f_wall)])









                    return f_wall

            def influencia(self, i):

                vel_media = np.array([0, 0])

                for j in range(n):

                    if 0 < norm(np.array(g_part[i].posicion) - np.array(g_part[j].posicion)) < r:
                        vel_media = vel_media + np.array(g_part[j].velocidad)

                if norm(vel_media) != 0:
                    vel_media = vel_media / norm(vel_media)

                return vel_media

            def objetivo(i, target):

                vel_target = (target - np.array(g_part[i].posicion)) / norm((target - np.array(g_part[i].posicion)))

                return vel_target

            class metodos():

                def __init__(self, n, g_part, vel, dt, alpha, tau):

                    self.n = n

                    self.theta_m = 0

                    # self.thetas = np.zeros(n)

                    self.s_x = 0

                    self.s_y = 0

                    self.vel = vel

                    self.v_act = 0

                    self.new_vel = np.zeros((n, 2))

                    self.new_vel[:, :] = 0

                def actualiza(self):

                    # actualiza velocidad y vector de posicion



                    for i in range(n):


                        fuerza = cc()

                        f_p_final = fuerza.f_particulas(i, A, B, k, kappa, r_p)

                        f_wall = fuerza.f_paredes(i, A, B, k, kappa, r_p)

                        vel_media = influencia(self, i)

                        vel_target = objetivo(i, target)

                        print('VEL MEDIA PARA ' + str(i) + ' es ' + str(vel_media))

                        print('F_wall para ' + str(i) + ' es ' + str(f_wall))

                        print('F_p para ' + str(i) + ' es ' + str(f_p_final))

                        self.new_vel[i, :] = (((1 - panic) * v_p * vel_target + panic * vel_media - g_part[
                            i].velocidad) / tau + f_wall + f_p_final) * dt + np.array(g_part[i].velocidad)

                        self.new_vel[i, :] = self.new_vel[i, :] / norm(self.new_vel[i, :])

                        print("neeeew", self.new_vel[i, :])

                        # print('LA velocidad en el eje x es')

                        # print(g_part[i].velocidad[0])

                        # print('LA velocidad en el eje y es')

                        # print(g_part[i].velocidad[1])



                        """Este cambio de new_vel viene porque antes se actualizaba cada punto

                        antes que los demas y en realidad deberia ser todos a la vez"""

                    for i in range(n):
                        g_part[i].velocidad = self.new_vel[i, :]

                        g_part[i].posicion = g_part[i].posicion + g_part[i].velocidad * dt

                        print("nnnnnn", g_part[i].velocidad)

                        matrizposicionx[t, i] = np.array(g_part[i].posicion[0])
                        matrizposiciony[t, i] = np.array(g_part[i].posicion[1])



                        # print('LA posicion en el eje x es')

                        # print(g_part[i].posicion[0])

                        # print('LA posicion en el eje y es')

                        # print(g_part[i].posicion[1])

            class visual():

                def __init__(self, g_part, indice):

                    self.i = indice

                    self.colores = ['.r', '.b', '.m', '.y', '.g', '*r', '*b', '*m', '*y', '*g']

                def dibuja(self):

                    plt.plot(g_part[i].posicion[0], g_part[i].posicion[1], self.colores[i])

                def datos_vuelta(self, t):

                    if i == 0:
                        print("------VUELTA " + str(t) + "------")

                    print("Velocidad " + str(i) + " : ", g_part[i].velocidad)

                    print("Posición " + str(i) + " : ", g_part[i].posicion)

                    print("Ángulo " + str(i) + " : ", np.degrees(g_part[i].angulo))

                def paredes(self):

                    if Tramo1.u_r[0] == 0 and Tramo1.u_r[1] == 1:

                        y1 = np.linspace(0, Tramo1.h_c[1], 10)

                        x1 = (Tramo1.u_r[0] / Tramo1.u_r[1]) * (y1 - Tramo1.p_r[1]) + Tramo1.p_r[0]

                        y2 = np.linspace(Tramo1.h_f[1], 50, 10)

                        x2 = (Tramo1.u_r[0] / Tramo1.u_r[1]) * (y2 - Tramo1.p_r[1]) + Tramo1.p_r[0]



                    else:

                        x1 = np.linspace(0, Tramo1.h_c[0], 10)

                        y1 = (Tramo1.u_r[1] / Tramo1.u_r[0]) * (x1 - Tramo1.p_r[0]) + Tramo1.p_r[1]

                        x2 = np.linspace(Tramo1.h_f[0], 50, 10)

                        y2 = (Tramo1.u_r[1] / Tramo1.u_r[0]) * (x2 - Tramo1.p_r[0]) + Tramo1.p_r[1]

                    plt.plot(x1, y1, 'k-', linewidth=3)

                    plt.hold(True)

                    plt.plot(x2, y2, 'k-', linewidth=3)



                    # class excel():

                    #   def __init__(self, n_col):



                    #      self.doc = openpyxl.load_workbook('DatosPy.xlsx')

                    #     self.hoja = self.doc.get_sheet_by_name('Hoja1')

                    #    self.n = n_col



                    # def modifica(self) :



                    #   self.hoja.append([self.n])

                    #  self.doc.save("DatosPy.xlsx")

            class particula():

                def __init__(self, v_o, r, pos, v):

                    self.radio = r

                    self.v_o = v_o



                    # self.v = v

                    # self.pos = pos

                def posicion(self):

                    self.posicion = pos

                    print("jeje")

                    # return self.posicion

                def velocidad(self):

                    self.velocidad = v / norm(v)

                    print("jiji")

                    # return self.velocidad

                def angulo(self):

                    if v[0] == 0:

                        self.angulo = np.pi / 2



                    else:

                        self.angulo = np.arctan(v[1] / v[0])

                    return self.angulo

            # lo del return no parece que haga na





            ####### PROGRAMA PRINCIPAL ########



            v_o = 1

            t = 0

            #n = 5

            r = 3

            r_w = 2

            dt = 1

            stop = t_in

            A = 2000

            B = 0.08

            r_p = 0.6

            kappa = 2.4 * (10 ** 5)

            masa = 80

            k = 1.2 * (10 ** 5)

            tau = 1

            alpha = 0.3  # Inercia a mantener la dirección del instante anterior

            target = np.array([25, 5])

            v_p = 1  # (nerviosos = 1.5 relajados = 0.6 y normales = 1)

            panic = 0.6  # parametro de panico

            n_col = 0

            dir_n = [0, 0]

            p = np.zeros(n)

            pos = np.zeros(n)

            v = np.zeros(n)

            matrizposicionx = np.empty((stop, n))
            matrizposiciony = np.empty((stop, n))

            g_part = []

            print(g_part)

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

                visual(g_part, i).dibuja()

            Tramo1 = pared(np.array([1, 0]), np.array([25, 25]), np.array([22, 25]), np.array([28, 25]))

            Tramo1.r()

            print(Tramo1.A, Tramo1.B, Tramo1.C)

            # print(g_part[0].posicion)

            # print(g_part[1].velocidad)

            plt.xlim(0, 50)

            plt.ylim(0, 50)

            fig = plt.figure()

            visual(g_part, i).paredes()

            plt.xlim(0, 50)

            plt.ylim(0, 50)

            # doc = openpyxl.load_workbook('DatosPy.xlsx')

            # hoja = doc.get_sheet_by_name('Hoja1')



            # with writer.saving(fig, "pruebafuerzas2.mp4", 100):

            for t in range(stop):

                Play = metodos(n, g_part, v_o, dt, alpha, tau)

                # print("Thetas med ", g_part.angulo)

                Play.actualiza()

                # print("x", g_part[0].posicion, g_part[1].posicion, g_part[2].posicion, g_part[3].posicion)

                # print("t",t)

                # print("tethas",np.degrees(g_part.angulo))



                for i in range(n):
                    visual(g_part, i).dibuja()

                    visual(g_part, i).datos_vuelta(t)

                    # print("tethas", np.degrees(g_part[i].angulo))

                    # writer.grab_frame()


            ##Aqui##

            for v in range(stop):

                for b in range(n):
                    matrizposicionx[v, b] = np.round(matrizposicionx[v, b], 2)
                    matrizposiciony[v, b] = np.round(matrizposiciony[v, b], 2)

            file = open('data2.json', 'w')
            file.write('[\n')
            for f in range(stop):
                line = '['
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




        if msg_tipo == 'atentado':

            import numpy as np

            import matplotlib.pyplot as plt

            from numpy.linalg import norm, det

            import math

            import matplotlib.animation as manimation

            import openpyxl

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

                        elif g_part[i].posicion[0] > self.right and matrizposiciony[t, i] != self.top and \
                                        matrizposiciony[
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

                                # plt.show()

                                # sys.exit()

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

            # plt.xlim(0, 50)
            # plt.ylim(0, 50)

            # fig = plt.figure()

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





        #print(self.connections)

        # [con.write_message('Hi!') for con in self.connections]
        self.write_message({"msg": message + ' al que lo manda'})


    def on_close(self):
        print('connection closed\n')



application = tornado.web.Application([(r'/', IndexHandler),(r'/ws', WSHandler),
(r'/(.*)', tornado.web.StaticFileHandler, {'path': r'E:\Descargas\DEFINITIVO\FINAL'})])

if __name__ == "__main__":
    http_server = tornado.httpserver.HTTPServer(application)
    http_server.listen(8888)
    tornado.ioloop.IOLoop.instance().start()