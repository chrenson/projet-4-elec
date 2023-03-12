import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

#Variables
L = 1 #Côté de la plaque carrée [m]
D = 10 #Distance émetteur-centre de la plaque [m]
R = 1 #distance émetteur-récepteur [m]
lambda0 = (3e8)/(2.4*10**9) #Longueur d'onde [m]
k = 2*np.pi/lambda0 #Nombre d'onde [rad/m]
eta  = 377 #Impédance du vide
l = 0.1 #Longueur du dipôle [m]
I = 1 #Courant dans le dipôle [A]
theta = -20 #angle de rotation de la plaque [°]
theta_arr = np.linspace(-90, 90, 6)
#Moment du dipôle 
p = np.array([0,0,I*l])


#Position de la plaque comme un nuage de points (dans un premier temps non-inclinée)
nx = 200 #nombre de points
x = np.linspace(-L/2,L/2,nx)
z = np.linspace(-L/2,L/2,nx)


dS = (L**2)/(nx**2) #surface d'un rectangle

y = np.linspace(0, D, nx)

H = np.zeros((nx,nx, 3),dtype = complex)
H_norm = np.zeros((nx,nx))
J = np.zeros((nx,nx, 3),dtype = complex)
J_norm = np.zeros((nx,nx))
I = np.zeros((nx,nx))
n = np.array([0,-1,0]) #normale à la plaque


#calcul de E incident, H incident et courant induit J

#matrice de rotation d'un angle theta selon l'axe z
plaque = np.zeros((nx,nx,3))
def calcul_J(theta):
    theta_rad = np.deg2rad(theta) #fct° sin et cos prennent des angles en rad
    rot = np.array([[np.cos(theta_rad), -np.sin(theta_rad), 0],[np.sin(theta_rad),np.cos(theta_rad),0],[0,0,1]])
    for i in range(nx):
        for m in range(nx):
            #on calcule u pour chaque point
            coor = np.array([x[m], 0, z[i]]) #coordonnées du point sur la plaque
            coor_rot = np.dot(coor, rot) #coordonnées du point incliné
            u = np.array([0, D, 0]) + coor_rot
            u_norm = np.linalg.norm(u)
            #vecteur u unitaire
            u_n = u/u_norm
            
            E = (1j*k*eta)/(4*np.pi)*np.cross(u_n, np.cross(u_n,p))*np.exp(-1j*k*u_norm)/u_norm
            
            Hv = np.cross(u_n,E)/eta
            
            n_rot = np.dot(n, rot)
            Jv = 2*np.cross(n_rot,Hv)
            
            H[i][m] = Hv
            H_norm[i][m]= np.linalg.norm(Hv)
            J[i][m] = Jv
            J_norm[i,m] = np.linalg.norm(Jv)
            I[i,m] = J_norm[i,m]*dS
            plaque[i][m] = u
    return J,J_norm,H, H_norm, I, plaque
        
#calcul du champ E diffracté en un point de l'espace
def E_diff(J,x,z, X,theta):
    #X = coordonnées du point de l'espace en lequel on calcule le champ E diffracté
    E_tot= np.zeros(3,dtype = complex)
    theta_rad = np.deg2rad(theta) #fct° sin et cos prennent des angles en rad
    rot = np.array([[np.cos(theta_rad), -np.sin(theta_rad), 0],[np.sin(theta_rad),np.cos(theta_rad),0],[0,0,1]])
    for i in range(nx):
        for m in range(nx):
            coor = np.array([x[m], 0, z[i]]) #coordonnées du point sur la plaque
            coor_rot = np.dot(coor, rot) #coordonnées du point incliné
            u = np.array([0, D, 0]) + coor_rot
            #u = np.array([x[m], D, z[i]])
            u2 = X-u
            u2_norm = np.linalg.norm(u2)
            u2_n = u2/u2_norm
            E_tot+= (1j*k*eta)/(4*np.pi)*np.cross(u2_n, np.cross(u2_n,J[i][m]))*dS*np.exp(-1j*k*u2_norm)/u2_norm
    return np.linalg.norm(E_tot)
            
#calcul du champ diffracté dans l'espace x,y
nr = 100
xs = np.linspace(-10*L/2,10*L/2,nr) #on prend un espace 10x plus grand que la plaque
ys = np.linspace(0, D-1, nr)
ysp = np.linspace(D, 1, nr) #distance entre la plaque et le point
def calcul_E_diff_tot(nr,xs,ys, ysp):
    E_diff_tot = np.zeros((nr,nr))
    for i in range(nr):
        for m in range(nr):
            X = np.array([xs[m],ys[i],0]) #coordonnées du point x,y et on prend z=0
            E_diff_tot[i][m] = E_diff(J,x,z,X,theta)
    return E_diff_tot 

#calcul de la section efficace
def section_eff(theta_arr):
    #E inc en (x,z) = (0,0)
    u = np.array([0,D,0])
    u_norm = np.linalg.norm(u)
    u_n = u/u_norm
    Ei = np.linalg.norm((1j*k*eta)/(4*np.pi)*np.cross(u_n, np.cross(u_n,p))*np.exp(-1j*k*u_norm)/u_norm)
    #champ réfléchi
    R_v = np.array([0,R,0])
    R2 = np.linalg.norm(R_v-u)
    sigma = np.zeros(len(theta_arr))
    for i in range (len(theta_arr)):
        Ji = calcul_J(theta_arr[i])[0]   
        Er = E_diff(Ji,x,z, R_v,theta_arr[i])
        sigma[i] = 4*np.pi*R2**2 * Er**2/Ei**2
    return sigma
        
#Affichage en 2D
def affichage_2D (H,x,z,title, xlabel, ylabel):
    plt.imshow(H, extent=(x[0],x[-1],z[0],z[-1]), cmap='viridis',origin='lower')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    titre = title + " (theta = " + str(theta) +"°)"
    plt.title(titre)
    plt.colorbar()
    plt.show()
#affichage de la plaque
def affichage_plaque():
    px = np.zeros(nx)
    py = np.zeros(nx)
    for i in range(nx):
        px[i] = plaque[0][i][0]
        py[i] = plaque[0][i][1]
    plt.plot(px,py)
    plt.show()

#J,J_norm,H, H_norm, I, plaque = calcul_J(theta)    
#affichage_2D (H_norm,x,z, 'Norme du champ H incident', 'x', 'z')
#affichage_2D (J_norm,x,z, 'Norme du champ J incident','x','z')
#affichage_2D (I,x,z,'Courant induit sur la plaque', 'x', 'z')
#affichage_plaque()

#E_diff_tot = calcul_E_diff_tot(nr,xs,ys, ysp)
#affichage_2D (E_diff_tot,xs,ysp, 'Norme du champ E diffracté', 'x', 'y')
sigma = section_eff(theta_arr)
plt.plot(theta_arr, sigma)
plt.title("section efficace en fonction de l'angle d'inclinaison")
plt.ylabel("section efficace [$m^2$]")
plt.xlabel("angle d'inclinaison [°]")
plt.show()

"""#3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(xs,ys,E_diff_tot,cmap='viridis')
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("E_diff_tot")
plt.show()"""