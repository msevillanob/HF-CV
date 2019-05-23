"""
===========HF-CV simulation through passivation parameters:==============
#                                                                       #
# The program was developed by Miguel Ángel Sevillano in the frame of   #
# the Master degree in Physics. If you have any question or suggestion  #
# please, contact to the author to the following emails:                #
# msevillanob@pucp.edu.pe or msevillano@uni.pe                          #
#-----------------------------------------------------------------------#
# If you want to simulate curves with comparison of experimental data   #
# you need to add three files in .txt with the names:                   #
# 1) cv_ideal_extracted.txt (the ideal HF-CV extracted from the HF-CV   #
# Messprogram)                                                          #
# 2) cv_experimental.txt (the experimental HF-CV curve)                 #
# 3) ditexperimental.txt (the experimentally extracted Dit)             #
#                                                                       #
# The program is constructed for SiO2/c-Si MOS system, if you want to   #
# use for others materials you need to change manually the constants    #
# associated with the material.                                         #
#-----------------------------------------------------------------------#
# Researchers should cite this work as follows:                         #
#                                                                       #
# Sevillano Bendezú, Miguel Ángel."Comparison and evaluation of         #		
# measured and simulated High-Frequency                                 #
# Capacitance-voltage curves of MOS structures for different interface  #
# passivation parameters". Master Thesis. 2019.                         #
#                                                                       #
#=======================================================================#
"""

import numpy as np
import scipy as sp
import matplotlib as mpl
import matplotlib.pyplot as plt           
import dit_tools.roots as rt
from scipy import integrate
from scipy import interpolate

def ditexperiment(et):
	filename = "ditexperimental.txt"
	data = np.loadtxt(filename)
	volexp = data[:,0]
	ditexp = data[:,1]

	dit = sp.interpolate.interp1d(volexp,ditexp)
	
	return dit(et)

def dit_gauss(phi):
	"""Gaussian model parameters"""
	Ae1 = 7.*10**15 #correct
	x01 = -0.56#correct
	td = 1/46#correct
	Ae2 = 2.59*10**14#correct
	x02 = 0.56#correct
	tg = 1/46 #correct
	Ag1 = 4.7*10**10#correct
	w1 = 0.24#correct
	xc1 = -0.375#correct
	Ag2 = 1.*10**10#correct
	w2 = 0.27#correct
	xc2 = -0.01#correct
	Ag3 = 2.5*10**10#correct
	w3 = 0.163#correct
	xc3 = 0.325#correct
	"""Gaussian model"""
	Dita = Ae1*np.e**(-abs(phi-x01)/(td))+(Ag1)/(w1*np.sqrt(np.pi/2.))*np.e**(-2.*(phi-xc1)**2/w1**2)+(Ag2)/(w2*np.sqrt(np.pi/2))*np.e**(-2*(phi-xc2)**2/w2**2)+(Ag3)/(w3*np.sqrt(np.pi/2))*np.e**(-2*(phi-xc3)**2/w3**2)+Ae2*np.e**(-abs(phi-x02)/(tg))
	return Dita
	
def dit_ushape(phi):
	"""U-shape model parameters"""
	Ditq0 = 4.5452631578947365*10**10 #degen sol3
	Ditc = 2.4315789473684212*10**14 #degen sol3
	Ditv = 1.5157894736842105*10**15 #degen sol3
	mc = 40
	mv = 40
	E0c = 0.56
	E0v = -0.56
	"""U-shape model"""
	Dit = Ditq0+Ditc*np.e**(-mc*abs(phi-E0c))+Ditv*np.e**(-mv*abs(phi-E0v))
	return Dit

########################################
def fermi(Etn,type,option):#*
	gd = 0.5
	#gd = 1.
	ga = 2.0
	#ga = 1.
	if option == -3:
		temp = gd
		gd = ga
		ga = temp
	
	if (type == "donor"):
		Etn = Etn/(K*T)
		f = (1./(1.+gd*np.e**(-Etn)))
		return f
	elif (type == "acceptor"):
		Etn = Etn/(K*T)
		f = (1./(1.+ga*np.e**(Etn)))
		return f
	
def phibulk(type): #V
	if (type == "p"):
		phib = (-(Kp*T)/charge)*np.log(Nd/ni)
		return phib
	elif (type == "n"):
		phib = ((Kp*T)/charge)*np.log(Nd/ni)
		return phib
	
	
def dimensionlessF(psi,type):#*
	psin = (charge*psi)/(Kp*T)
	phibn = (charge*phibulk(type))/(Kp*T)
	F = np.sqrt(2.)*np.sqrt((-psin)*np.sinh(phibn)-np.cosh(phibn)+np.cosh(psin+phibn))
	return F

def Qsemic(psi,type):#cm^-2
	Qs = np.sign(-psi)*((e0*esi)/(lambi))*((Kp*T)/(charge))*dimensionlessF(psi,type)
	return Qs

def Qit(psi,type,option):#cm^-2					
	phi = psi + phibulk(type)
	if optional == 2:
		filename = "ditexperimental.txt"
		data = np.loadtxt(filename)
		philexp = data[:,0]
		ditexp = data[:,1]


		if philexp[-1]<philexp[0]:
			El = philexp[-1]
			Eh = philexp[0]
		else:
			Eh = philexp[-1]
			El = philexp[0]
	elif optional ==1:
		El = -0.56
		Eh = 0.56

	if option == -3:
		Qitd, er1 = sp.integrate.quad(lambda Es: (dit_ushape(-Es))*fermi(Es-phi,"donor",option),-Eh,0)
		Qita, er2 = sp.integrate.quad(lambda Es: (dit_ushape(-Es))*fermi(Es-phi,"acceptor",option),0,-El)
	else: 
		Qitd, er1 = sp.integrate.quad(lambda Es: (dit_ushape(Es))*fermi(Es-phi,"donor",option),El,0)
		Qita, er2 = sp.integrate.quad(lambda Es: (dit_ushape(Es))*fermi(Es-phi,"acceptor",option),0,Eh)	
		
	Qit = Qitd-Qita
	return Qit

def theoreticalU(psi,type,option):#V

	if (option == -1):
		U = psi-(Qsemic(psi,type)*area/cox)-wms+phibulk(type)			
		return U
	elif (option == -2 ):
		U = psi-(Qsemic(psi,type)*area/cox)-wms+phibulk(type)+(Nox*charge*area/cox)
		return U
	elif (option == -3):
		U = psi-(Qsemic(psi,type)*area/cox)-wms+phibulk(type)+((Nox*charge)*area/cox)-(charge*Qit(psi,type,option))*area/cox
		return U
		
		
		
	elif (option == 1):
		U = psi-(Qsemic(psi,type)*area/cox)+wms+phibulk(type)
		return U
	elif (option == 2):
		U = psi-(Qsemic(psi,type)*area/cox)+wms+phibulk(type)-(Nox*charge*area/cox)
		return U
	elif (option == 3):
		U = psi-(Qsemic(psi,type)*area/cox)+wms+phibulk(type)-((Nox*charge)*area/cox)-(charge*Qit(psi,type,option))*area/cox
		return U

def bandbending(gatevolt,type,option): #V
	Us = lambda psi: theoreticalU(psi,type,option)-gatevolt
	phin = rt.bisection(Us,-10,10,10**(-10))
	
	if type == "p":
		if phin<=psiLm:
			mp = 0 
		else:
			mp = 1			
			
	elif type == "n":
		if phin>=psiLm:
			mp = 0
		else:
			mp = 1
			
	return phin, mp


def capacitanceL(gatevolt,type,option):
	psi, mp = bandbending(gatevolt,type,option)
	if mp == 1:
		psi = psiLm
		
	psin = charge*psi/(Kp*T)


	if (type == "p"):
		
		if psin == 0:
			CL = area*Cfbs
			
		else:
			CL = (area/np.sqrt(2.))*np.sign(psi)*Cfbs*((1.-np.e**(-psin))/np.sqrt((psin)-1.+np.e**(-psin)))
		
		
		return CL
	elif (type == "n"):
		if psin == 0:
			CL = area*Cfbs
		else:
			CL = (area/np.sqrt(2.))*np.sign(psin)*Cfbs*((np.e**(psin)-1.)/np.sqrt(-(psin)-1.+np.e**(psin)))
	
		return CL

def totalCap(gatevolt,type,option):
	CtL = 1./((1./cox)+(1./(capacitanceL(gatevolt,type,option))))
	return CtL/area #NORMALIZED OR NOT
	
	
def capacitanceLowf(psi,type):
	psin = charge*psi/(Kp*T)
	phib = charge*phibulk(type)/(Kp*T)
	if psin == 0:
		csL = area*Cfbs
	else:
		csL = area*(e0*esi/lambi*(np.sign(psin))*(np.sinh(psin+phib)-np.sinh(phib))/np.sqrt(2.*((-psin)*np.sinh(phib)-np.cosh(phib)+np.cosh(psin+phib))))
	
	cL = 1./((1./cox)+(1./csL))
	cL = cL/cox
	
	return cL

def integra(psi1,psi2,type):

	if psi1<-phibulk(type):
		psi1 = -phibulk(type)
				 
	psin1 = charge*psi1/(Kp*T)
	psin2 = charge*psi2/(Kp*T)				
	inte, err =sp.integrate.quad(lambda psin: (1.-np.exp(-psin))*(np.exp(psin)-psin-1.)/(2.*dimensionlessF(Kp*T*psin/charge,type)**3),psin1,psin2)
	
	return inte

def exactsolHF(gatevolt,type,option):
	if type == "p":
		flag = 0
	elif type =="n":
		flag = 1
	type = "p"
	psi=np.zeros(len(gatevolt))
	cH=np.zeros(len(gatevolt))
	phb = charge*np.abs(phibulk(type))/(Kp*T)
	inte = 0
		
	for j in range(0,len(gatevolt)):
		if flag == 0:
			i = j
			psi[i], mp = bandbending(gatevolt[i],type,option)
		elif flag == 1:		
			i = -1-j
			option = -np.abs(option)
			psi[i], mp = bandbending(-gatevolt[i],type,option)

		psin = charge*psi[i]/(Kp*T)
		if psin<phb:
			cH[i] = capacitanceLowf(psi[i],type)

		else:
			if flag == 0:
				inte = inte+integra(psi[i-1],psi[i],type)
			elif flag ==1:
				inte = inte+integra(psi[i+1],psi[i],type)

			delta = (np.exp(psin)-psin-1.)/(dimensionlessF(psi[i],type)*np.exp(phb)*inte)
			
			deno = np.exp(phb)*(1.-np.exp(-psin))+np.exp(-phb)*((np.exp(psin)-1.)/(1.+delta))
			
			WH = lambi*(2.*dimensionlessF(psi[i],type))/deno
			cH[i] = 1./(1.+(eox*WH)/(esi*dox))
			cH[i] = cH[i]

	return cH*cox/area
	

##*****Data text files******##

file1 = open('CV_real.txt','w')
file2 = open('CVtheo_op1','w')
file3 = open('Cverrortheo','w')
file4 = open('CVtheo_op3','w')
file5 = open('CVtheo_op1_exact','w')
file6 = open('CVtheo_op3_exact','w')
file8 = open('CVtheo_op2','w')
file9 = open('CVtheo_op2_exact','w')
file10= open('correctionfactor','w')
file11= open('ditsimulated','w')
file12= open('3.Vg_vs_phi','w')
file13= open('Qit','w')
file14= open('Qsc','w')
file15= open('Qoxeff','w')
#explanation
file16= open('Fermidon','w')
file17= open('Fermiace','w')
file18= open('Ditdo','w')
file19= open('Qpositive','w')
file20= open('Qnegative','w')
file21= open('Ditac','w')

print("HF-CV simulation".capitalize().center(80,"="))
print("Select one choice:\n \t 1.Theoretical CV curve included work function difference\n\t 2.CV curve with Qox,eff effect\n\t 3.CV curve with Qox,eff and Dit effects\n")
print("".center(80,"-"))
option =int(input())
print("".center(80,"-"))
print("Do yo want to maintain parameters or change the values by default on the script? 1/2\n")
param = int(input())

###############################
#######"""Constants"""#########
###############################
charge = 1.602176462*10**-19 #C
Kp = 1.3806503*10**-23 #J*K^-1
K = 8.6173303*10**(-5)
e0 = 8.85418781762*10**-14#F*cm^-1
esi = 11.8 #*
eox = 3.8 #*
Ec = 0.56 #ev
Ev = -0.56 #eV
Nv = 1.06*10**19 #cm^-3
Nc = 2.84*10**19 #cm^-3

###############################
#######"""parameters"""########
###############################

if param == 1:

	type = "n"
	T = 296.5 #K
	Nd = 1.341158*10**15#cm^-2
	dox = 100.6521*10**-7#cm
	Eg = 1.12 #eV
	phim = 4.1 #eV
	Xsi = 4.05 #eV
	Nox = 9.963367*10**11 #cm^-2
	diam = 0.615000*10*(-1) #cm
	ni = 9.15*10**19*((T/300)**2)*np.e**(-6880/T)
	DitMG = 2.488396*10**10 #cm^-2 eV^-1
	#################################
	wms = -0.22 #V
elif param == 2:

	type = input("Select the semiconductor type (p or n):\n")
	T = float(input("Input the temperature in K (e.g. 300):\n"))
	Nd = float(input("Input doping concentration in cm^-2 (e.g. 1E15):\n"))
	dox = float(input("Input the oxide thickness in nm (e.g. 100):\n"))*10**-7#cm
	Eg = 1.12 #eV
	phim = 4.1 #eV
	Xsi = 4.05 #eV
	Nox = float(input("Input the Nox (effective oxide charges density)in cm^-2 (e.g. 1E12):\n"))
	diam = float(input("Input the dot contact diameter (e.g. 0.5):\n")) #cm
	ni = 9.15*10**19*((T/300)**2)*np.e**(-6880/T)
	DitMG = 2.488396*10**10 #cm^-2 eV^-1
	#################################
	wms = -0.22 #V

optional = int(input("Do you want to simulate a HF-CV curve base on the above parameters or experimental data? 1/2?:\n"))
if optional == 1:
	vlow = float(input("Enter the lowest voltage step (e.g. -10):\n")) 
	vhigh = float(input("Enter the highest voltage step(e.g. 10):\n")) 


area = (np.pi/2.)*(diam/2.)**2
cox = (e0*eox)*area/dox
#Debye length
lambi = np.sqrt((e0*esi*Kp*T)/(2*(charge**2)*ni)) #cm
lambe = np.sqrt((e0*esi*Kp*T)/((charge**2)*Nd)) #cm
#%%matchpoint calculation
 
vLm = 2*charge*np.abs(phibulk(type))/(Kp*T)

while True:
	vL=vLm
	vLm = 2*charge*np.abs(phibulk(type))/(Kp*T)+np.log(vL+1.25)-2.25	
	err = np.abs(1-vL/vLm)	
	if err<=10**(-5):				
		break

vlm2 = 2.1*charge*np.abs(phibulk(type))/(Kp*T)+1.33	
psiLm = (Kp*T)*vLm/charge

if type == "n":
	psiLm = -psiLm


###########Flatband and Midgap voltage################
psflatb = 0
Vflatb = theoreticalU(psflatb,type,option)
print ("Flatband voltage (in V):%s\n"%(Vflatb))

psmidg = -phibulk(type)
Vmidg = theoreticalU(psmidg,type,option)
print ("Midgap voltage (in V):%s\n"%(Vmidg))	
	
Cfbs = e0*esi/lambe	
if optional == 2:
	#############DATA FOR FITING#####################
	filename = "ditexperimental.txt"
	data = np.loadtxt(filename)
	philexp = data[:,0]
	ditexp = data[:,1]

	ditmod = np.zeros(len(philexp))
	ditmod2 = np.zeros(len(philexp))
	for i in range (0, len(philexp)):
		ditmod[i] = dit_gauss(philexp[i])
		ditmod2[i] = dit_ushape(philexp[i])

	for i in range (0, len(philexp)):
		file11.write(np.str(philexp[i]))
		file11.write(' ')
		file11.write(np.str(ditmod2[i]))
		file11.write('\n')	

	fig, ax = plt.subplots()
	plt.xlabel('Energy (eV)')
	plt.ylabel('Dit (cm^-2eV^-1)')
	plt.title('Dit machtching')
	ax.semilogy(philexp,ditexp,'bs')
	ax.semilogy(philexp,ditmod,'r^')
	ax.semilogy(philexp,ditmod2,'g^')
	ax.grid()
	plt.show()

	Nit = np.trapz(ditmod2,philexp)
	Nit2 = np.trapz(ditexp,philexp)
	print("Dit average modeled U-shape/10**11:")
	print((abs(Nit/10**11))/(philexp[0]-philexp[-1]))
	print("Dit average experimental/10**11:")
	print((abs(Nit2/10**11))/(philexp[0]-philexp[-1]))
	#############################################

	#GRAPH QIT
	####################
	qit = np.zeros(len(philexp))
	for i in range (0,len(philexp)):
		qit[i] = Qit(philexp[i]-phibulk(type),type,option)
	fig, ax = plt.subplots()
	ax.semilogy(philexp,abs(qit))
	ax.grid()
	plt.show()

	for i in range (0, len(philexp)):
		file13.write(np.str(philexp[i]-phibulk(type)))
		file13.write(' ')
		file13.write(np.str(abs(charge*qit[i])))
		file13.write('\n')	

	qitt=abs(qit)
	qitaverage = np.trapz(qitt,philexp)
	qitaverage = qitaverage/(abs(philexp[0]-philexp[-1]))
	print("qitaverage/10**10:")
	print(abs(qitaverage)/10**10)
	print("__________")

	#GRAPH QSEMIC
	#########################
	qsimx = np.zeros(len(philexp))
	for i in range (0, len(philexp)):
		qsimx[i] = np.abs(Qsemic(philexp[i]-phibulk(type), type))

	fig, ax = plt.subplots()
	ax.semilogy(philexp,qsimx)
	ax.grid()
	plt.show()

	for i in range (0, len(philexp)):
		file14.write(np.str(philexp[i]-phibulk(type)))
		file14.write(' ')
		file14.write(np.str(abs(qsimx[i])))
		file14.write('\n')

	for i in range (0, len(philexp)):
		file15.write(np.str(philexp[i]-phibulk(type)))
		file15.write(' ')
		file15.write(np.str(charge*Nox))
		file15.write('\n')	



	##########Error###############################
	filename1 = "cv_experimental.txt"
	data1 = np.loadtxt(filename1)
	biasexp = data1[:,0]
	capaexp = data1[:,1]

	if (option == 1):
		capmod = np.zeros(len(biasexp))
		error = np.zeros(len(biasexp))
		errorexa = np.zeros(len(biasexp))

		capmod_ex = exactsolHF(biasexp,type,option)
		
		
		for i in range(0,len(biasexp)):
			capmod[i] = totalCap(biasexp[i],type,option)
		
		for i in range(0,len(biasexp)):
			error[i] = 100.*abs(capaexp[i]-capmod[i])/capaexp[i]
			errorexa[i] = 100.*abs(capaexp[i]-capmod_ex[i])/capaexp[i]
		
		plt.plot(biasexp,error)
		plt.show()
		
		
		for i in range (0, len(biasexp)):
			file2.write(np.str(biasexp[i]))
			file2.write(' ')
			file2.write(np.str(capmod[i]))
			file2.write('\n')

		for i in range (0, len(biasexp)):
			file3.write(np.str(biasexp[i]))
			file3.write(' ')
			file3.write(np.str(errorexa[i]))
			file3.write('\n')
			
		for i in range (0, len(biasexp)):
			file5.write(np.str(biasexp[i]))
			file5.write(' ')
			file5.write(np.str(capmod_ex[i]))
			file5.write('\n')		

		phimodel = np.zeros(len(biasexp))

		for i in range (0, len(biasexp)):

			phimodel[i],mp = bandbending(biasexp[i],type,option)
			phimodel[i] = phimodel[i]+phibulk(type)
			
			
		for i in range (0, len(phimodel)):
			file12.write(np.str(phimodel[i]))
			file12.write(' ')
			file12.write(np.str(biasexp[i]))
			file12.write('\n')	
				
	##########################		
	"""Correction CV_exact"""
	##########################		

	filename = "cv_ideal_extracted.txt"
	data1 = np.loadtxt(filename)
	biasexp = data1[:,0]
	capaexp = data1[:,1]

	filenamecorrect = "cv_experimental.txt"
	datacorrect = np.loadtxt(filenamecorrect)
	biasexpcor = datacorrect[:,0]
	capaexpcor = datacorrect[:,1]
	differenceerr = np.outer(np.zeros(len(biasexpcor)),np.zeros(2))
	capmod_excor = exactsolHF(biasexpcor,type,1)
	if (option == 2):
		
		capmod = np.zeros(len(biasexp))
		error = np.zeros(len(biasexp))
		capmod_ex = exactsolHF(biasexp,type,option)
		print ("Are you really sure that the theoretical calculation by the analyzer is correct? y/n?\n")
		ans = input()
		if ans == "y":
			####Correction by theoretical curve#####
			for i in range(0,len(biasexpcor)):
				differenceerr[i][1] = abs(capaexpcor[i]-capmod_excor[i])
				if capaexpcor[i]>capmod_excor[i]:
					differenceerr[i][0] = 1
				elif capaexpcor[i]==capmod_excor[i]:
					differenceerr[i][0] = 0
				else: 
					differenceerr[i][0] =-1		
					
				capmod_ex[i] = capmod_ex[i]+differenceerr[i][0]*differenceerr[i][1]

		for i in range(0,len(biasexp)):
			capmod[i] = totalCap(biasexp[i],type,option)
		for i in range(0,len(biasexp)):
			error[i] = 100.*abs(capaexp[i]-capmod_ex[i])/capaexp[i]
		
		plt.plot(biasexp,error)
		plt.show()

		for i in range (0, len(biasexp)):
			file8.write(np.str(biasexp[i]))
			file8.write(' ')
			file8.write(np.str(capmod[i]))
			file8.write('\n')

		for i in range (0, len(biasexp)):
			file3.write(np.str(biasexp[i]))
			file3.write(' ')
			file3.write(np.str(error[i]))
			file3.write('\n')
			
		plt.plot(biasexp,error)
		plt.show()		

			
		for i in range (0, len(biasexp)):
			file9.write(np.str(biasexp[i]))
			file9.write(' ')
			file9.write(np.str(capmod_ex[i]))
			file9.write('\n')
				

	filename2 = "cv_ideal_extracted.txt"
	data2 = np.loadtxt(filename2)
	biasexp = data2[:,0]
	capaexp = data2[:,1]

	if (option == 3):
		capmod = np.zeros(len(biasexp))
		error = np.zeros(len(biasexp))
		capmod_ex = exactsolHF(biasexp,type,option)
		
		print ("Are you really sure that the theoretical calculation by the analyzer is correct? y/n?\n")
		ans = input()
		if ans == "y":
			####Correction by theoretical curve#####
			for i in range(0,len(biasexpcor)):
				differenceerr[i][1] = abs(capaexpcor[i]-capmod_excor[i])
				if capaexpcor[i]>capmod_excor[i]:
					differenceerr[i][0] = 1
				elif capaexpcor[i]==capmod_excor[i]:
					differenceerr[i][0] = 0
				else: 
					differenceerr[i][0] =-1	
					
				capmod_ex[i] = capmod_ex[i]+differenceerr[i][0]*differenceerr[i][1]
		
		
		for i in range(0,len(biasexp)):
			capmod[i] = totalCap(biasexp[i],type,option)
		for i in range(0,len(biasexp)):
			error[i] = 100.*abs(capaexp[i]-capmod_ex[i])/capaexp[i]
		
		plt.plot(biasexp,error)
		plt.show()
		
		for i in range (0, len(biasexp)):
			file3.write(np.str(biasexp[i]))
			file3.write(' ')
			file3.write(np.str(error[i]))
			file3.write('\n')	
			
		for i in range (0, len(biasexp)):
			file4.write(np.str(biasexp[i]))
			file4.write(' ')
			file4.write(np.str(capmod[i]))
			file4.write('\n')
		for i in range (0, len(biasexp)):
			file6.write(np.str(biasexp[i]))
			file6.write(' ')
			file6.write(np.str(capmod_ex[i]))
			file6.write('\n')

		phimodel = np.zeros(len(biasexp))

		for i in range (0, len(biasexp)):

			phimodel[i],mp = bandbending(biasexp[i],type,option)
			phimodel[i] = phimodel[i]+phibulk(type)
			
			
		for i in range (0, len(phimodel)):
			file12.write(np.str(phimodel[i]))
			file12.write(' ')
			file12.write(np.str(biasexp[i]))
			file12.write('\n')	

	#################################################
	for i in range (0, len(gate)):
		file1.write(np.str(gate[i]))
		file1.write(' ')
		file1.write(np.str(cap[i]))
		file1.write('\n')

	phisim = np.linspace(-0.56,0.56,200)
	ditmoac = np.zeros(len(phisim))
	ditmodo = np.zeros(len(phisim))
	fermidon = np.zeros(len(phisim))
	fermiace = np.zeros(len(phisim))
	qposi= np.zeros(len(phisim))
	qnega = np.zeros(len(phisim))
	phicase = 0.3
	for i in range (0, len(phisim)):
		file16.write(np.str(phisim[i]))
		file16.write(' ')	
		file17.write(np.str(phisim[i]))
		file17.write(' ')
		file18.write(np.str(phisim[i]))
		file18.write(' ')
		file19.write(np.str(phisim[i]))
		file19.write(' ')
		file20.write(np.str(phisim[i]))
		file20.write(' ')
		file21.write(np.str(phisim[i]))
		file21.write(' ')
		

		if phisim[i]<0:
			fermidon[i]=fermi(phisim[i]-phicase,'donor',option)
			ditmodo[i] = dit_gauss(phisim[i])
			qposi[i] = fermidon[i]*ditmodo[i]
			file16.write(np.str(fermidon[i]))
			file16.write('\n')
			file19.write(np.str(qposi[i]))
			file19.write('\n')
			file18.write(np.str(ditmodo[i]))
			file18.write('\n')	
			j = 0
		else:
		
			fermiace[j]=fermi(phisim[i]-phicase,'acceptor',option)
			ditmoac[j] = dit_gauss(phisim[i])
			qnega[j] = fermiace[j]*ditmoac[j]
			file17.write(np.str(fermiace[j]))
			file17.write('\n')
			file20.write(np.str(qnega[j]))
			file20.write('\n')
			file21.write(np.str(ditmoac[j]))
			file21.write('\n')			
			j = j+1


#Capacitane curves: Closed form and exact solution
###############################################
elif optional == 1:
	gate = np.linspace(vlow,vhigh,100)
	cap = np.zeros(len(gate))
	for i in range (0, len(gate)):
		cap[i] = totalCap(gate[i],type,option)

	#plt.plot(gate,cap)
	capexa = exactsolHF(gate,type,option)

	#plt.plot(gate,capexa,"*")
	#plt.show()

	fig, ax = plt.subplots()
	ax.plot(gate, cap,"o",label="Closed-form approx.")
	ax.plot(gate, capexa,"*",label="Exact solution")
	ax.legend()
	ax.set(xlabel='Gate Voltage (V)', ylabel='C (F/cm^2)',
		title='HF-CV simulation')
	ax.grid()

	fig.savefig("HF-CV_Sim.png")
	plt.show()
	




file1.close()
file2.close()
file3.close()
file4.close()
file5.close()
file6.close()
file8.close()
file9.close()
file10.close()
file11.close()
file12.close()
file13.close()
file14.close()
file15.close()
file16.close() #fermidon
file17.close() #fermiace
file18.close() #Ditdo
file19.close() #Qpositive
file20.close() #Qnegative
file21.close() #Ditac