import numpy as np

goal = 11

# how many points in graph? x points
MATRIX_SIZE = 16+1
gamma = 0.8

cc = 2
I = -1*cc
D  = +1*cc
Ab = -1*cc
Ar = +1*cc
As = 0*cc

rg = +5
rr = -5
# create matrix x*y
R = np.matrix(np.ones(shape=(MATRIX_SIZE, MATRIX_SIZE)))
R *= -3

R[1,1]= As
R[1,2]= D

R[2,2]= As
R[2,1]= I
R[2,3]= D

R[3,2]= I
R[3,4]= D
R[3,3]= As

R[4,3]= I
R[4,8]= Ab
R[4,4]= As

R[8,4]= Ar
R[8,7]= I
R[8,12]= Ab
R[8,8]= As

R[7,8]= D
R[7,6]= I#D
R[7,11]= Ab #
R[7,7]= As

R[6,6]= As
R[6,7]= D#I
R[6,5]= I #D
R[6,10]= Ab#  A

R[5,5]= rr #  A
R[5,6]= rr #I
R[5,9] = rr #A

R[9,5]= Ar # Arriba
R[9,9]= As #
R[9,10] = D #I   
R[9,13] = Ab #ABajo

R[10,6] = Ar #Ar
R[10,9] = I #D
R[10,11] = D #I
R[10,10] = As #I

R[11,7] = Ar#A  
R[11,12] = D #I
R[11,10] = I #D
R[11,11] = Ab #I

R[12,12] = rg #
R[12,8] = rg # AR
R[12,11] = rg #D

R[13,13] = As #I
R[13,9] = Ar #Ar
R[13,14] = D #I

R[14,13] = I #D
R[14,15] = D #I
R[14,14]= As

R[15,15] = As
R[15,16] = D#  I
R[15,14] = I # D

R[16,16] = As
R[16,15] = I#

#print(R)

Q = np.matrix(np.zeros([MATRIX_SIZE,MATRIX_SIZE]))

# learning parameter
#print(Q)

initial_state = 2


def available_actions(state):
    current_state_row = R[state,]
    av_act = np.where(current_state_row >= -2)[1]
    return av_act

available_act = available_actions(initial_state) 



def sample_next_action(available_actions_range):
	next_action = int(np.random.choice(available_actions_range,1))
	return next_action

action = sample_next_action(available_act)


c = 4
def update(current_state, action, gamma):
    
  max_index = np.where(Q[action,] == np.max(Q[action,]))[1]

  
  if max_index.shape[0] > 1:
      max_index = int(np.random.choice(max_index, size = 1))
  else:
      max_index = int(max_index)
  max_value = Q[action, max_index]
  
  Q[current_state, action] = R[current_state, action] + gamma * max_value
  print('max_value', R[current_state, action] + gamma * max_value +c) 
  
  if (np.max(Q) > 0):
    return(np.sum(Q/np.max(Q)*100))
  else:
    return (0)
    
update(initial_state, action, gamma)



# Training
scores = []
for i in range(700):
    current_state = np.random.randint(0, int(Q.shape[0]))
    print("current_state",current_state)
    available_act = available_actions(current_state)
    print("available_act",available_act)
    if len(available_act)>0:
    	action = sample_next_action(available_act)
    else:
    	action = current_state
    print("action",action)
    score = update(current_state,action,gamma)
    scores.append(score)
    print ('Score:', str(score))
    print("\n")
    
print("Trained Q matrix:")
print(Q/np.max(Q)*100)


# Testing
current_state = 1
objetivo = 12
steps = [current_state]
print("\n")
print("   -------------------------")
print("   inicio :", current_state , "objetivo :", objetivo)
print("     ")
print("   Rs: ","Arriba ",Ar," Abajo ",Ab," Izquierda ",I, "Derecha ",D)
print("   Rg: ",rg )
print("   Rr: ",rr )
print("   gamma :",gamma)
print("   constante :",c)
print("   -------------------------")

while current_state != objetivo:

    next_step_index = np.where(Q[current_state,] == np.max(Q[current_state,]))[1]
    #print(next_step_index)
    
    if next_step_index.shape[0] > 1:
        next_step_index = int(np.random.choice(next_step_index, size = 1))
    else:
        next_step_index = int(next_step_index)
    
    steps.append(next_step_index)
    current_state = next_step_index

print("   Mas eficiente path:")
print("   ",steps)
print("\n")

