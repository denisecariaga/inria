#! /usr/bin/env python3
# _*_ coding: utf8 _*_

from math import sqrt
import numpy as np
import matplotlib.pyplot as plt

###  PARAMETRES  ###
### VARIABLES  ###
R = 0.1 #rayon de la bille
m = 1 #masse de la bille en kg
vo = 0 #vitesse initiae de la bille en m.s-1
qo = 1 #position initiale du centre de la bille en m
muo = 0 #impulsion au temps initial en N.s
h = 0.001 #pas de temps en s
e = 0.7 #coefficient de restitution sans unite

###  FIXES  ###
g = 10 #acceleration pesanteur en m.s-2
to = 0 #temps initial en s
tf = 5 #temps final en s

###  DUREE DE CHUTE tc ###
###  permet de definir ordre de grandeur duree chute  ###
D = (vo*vo)-(4*(-g)*qo)
if D > 0:
    t1=(-vo-sqrt(D))/(-2*g)
    t2=(-vo+sqrt(D))/(-2*g)
    tc=max(t1, t2)
if D == 0 :
    tc = vo/(2*g)

print(tc)
###  CALCUL DES VITESSES ET POSITIONS DE LA BILLE  ###
### INITIALISATION  ###
t = [to] #vecteur temps
v = [vo] #vecteur vitesse
q = [qo] #vecteur position
mu= [0]  #vecteur impulsion
i = 1    #iteration pour reperage dans les tableaux

###  BOUCLE SUR LE TEMPS  ###
#while t[-1] <= (15*tc): #tant que le dernier pas de temps calcule est inferieur ou egal a x fois le temps de chute tc initial
while t[-1] <= (tf):
   t.append(t[i-1]+h) #ajout nouveau pas de temps au vecteur t
   mu.append(0) #ajout de la nouvelle impulsion
   v.append(v[i-1]-h*g) #ajout de la nouvelle vitesse
   q.append(q[i-1]+h*v[i]) #ajout de la nouvelle position
   if q[i-1]-R<=0: #contact suppose
       if (v[i]+e*v[i-1])<=0: #verification du contact depasse
           v[i] = -e*v[i-1] #inversion de la vitesse moyennant ceff rest
           mu[i] = h*m*g+m*(1+e)*v[i] #calcul de l'impulsion de contact
           q[i]=q[i-1]+h*v[i] #mise a jour de la position en i fausse
           ''' print('v',v[i])
           print('v-1',v[i-1])
           print('mu',mu[i])
           print('q',q[i])
           print('q-1',q[i-1])
           print('t',t[i])
           input()'''
   i = i+1 #passage au pas de temps suivant


    
###  POST PROCESSING  ###

###  CALCUL ENERGIES  ###
###  CINETIQUE  ###
ec = []
i=0
for elem in v:
    ec.append((1/2)*m*v[i]*v[i])
    i=i+1
    
###  POTENTIELLE  ###
ep = []
i=0
for elem in q:
    ep.append(m*g*q[i])
    i=i+1

###  TOTALE  ###
et = []
i=0
for elem in ec:
    et.append(ec[i]+ep[i])
    i=i+1


###  TRACE COURBES  ###

fig1 = plt.figure()
ax1 = plt.axes()
plt.plot(t,q)

ax1.set_xlim(0, tf)
ax1.set_ylim(0, qo)
ax1.set(xlabel='temps (s)', ylabel='vitesse (m/s)')

fig2 = plt.figure()
ax2 = plt.axes()
plt.plot(t,ec)
plt.plot(t,ep)
plt.plot(t,et)
ax2.set(xlabel='temps (s)', ylabel='energies(J)')

fig3 = plt.figure()
ax3 = plt.axes()
plt.plot(t,v)
#plt.plot(t,mu)
ax1.set(xlabel='temps (s)', ylabel='vitesse (m/s)')
plt.show()





